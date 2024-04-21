#!/usr/bin/env python
"""Run affiliate server.

Usage:

start affiliate-server using dev webserver:

    ./scripts/affiliate_server.py openlibrary.yml 31337

start affiliate-server as fastcgi:

    ./scripts/affiliate_server.py openlibrary.yml fastcgi 31337

start affiliate-server using gunicorn webserver:

    ./scripts/affiliate_server.py openlibrary.yml --gunicorn -b 0.0.0.0:31337


Testing Amazon API:
  ol-home0% `docker exec -it openlibrary-affiliate-server-1 bash`
  openlibrary@ol-home0:/openlibrary$ `python`

```
import web
import infogami
from openlibrary.config import load_config
load_config('/olsystem/etc/openlibrary.yml')
infogami._setup()
from infogami import config;
from openlibrary.core.vendors import AmazonAPI
params=[config.amazon_api.get('key'), config.amazon_api.get('secret'),config.amazon_api.get('id')]
web.amazon_api = AmazonAPI(*params, throttling=0.9)
products = web.amazon_api.get_products(["195302114X", "0312368615"], serialize=True)
```
"""

import itertools
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import date

import web

import _init_path  # noqa: F401  Imported for its side effect of setting PYTHONPATH

import infogami
from infogami import config
from openlibrary.config import load_config as openlibrary_load_config
from openlibrary.core import cache, stats
from openlibrary.core.imports import Batch
from openlibrary.core.vendors import AmazonAPI, clean_amazon_metadata_for_load
from openlibrary.utils.dateutil import WEEK_SECS
from openlibrary.utils.isbn import (
    normalize_isbn,
    isbn_13_to_isbn_10,
    isbn_10_to_isbn_13,
)

logger = logging.getLogger("affiliate-server")

# fmt: off
urls = (
    '/isbn/([0-9X-]+)', 'Submit',
    '/status', 'Status',
    '/clear', 'Clear',
)
# fmt: on

API_MAX_ITEMS_PER_CALL = 10
API_MAX_WAIT_SECONDS = 0.9
AZ_OL_MAP = {
    'cover': 'covers',
    'title': 'title',
    'authors': 'authors',
    'publishers': 'publishers',
    'publish_date': 'publish_date',
    'number_of_pages': 'number_of_pages',
}

batch_name = ""
batch: Batch | None = None

web.amazon_queue = queue.Queue()  # a thread-safe multi-producer, multi-consumer queue
web.amazon_lookup_thread = None


def get_current_amazon_batch() -> Batch:
    """
    At startup or when the month changes, create a new openlibrary.core.imports.Batch()
    """
    global batch_name, batch
    if batch_name != (new_batch_name := f"amz-{date.today():%Y%m}"):
        batch_name = new_batch_name
        batch = Batch.find(batch_name) or Batch.new(batch_name)
    assert batch
    return batch


def get_isbns_from_book(book: dict) -> list[str]:  # Singular: book
    return [str(isbn) for isbn in book.get('isbn_10', []) + book.get('isbn_13', [])]


def get_isbns_from_books(books: list[dict]) -> list[str]:  # Plural: books
    return sorted(set(itertools.chain(*[get_isbns_from_book(book) for book in books])))


def is_book_needed(book: dict, edition: dict) -> list[str]:
    """
    Should an OL edition's metadata be updated with Amazon book data?

    :param book: dict from openlibrary.core.vendors.clean_amazon_metadata_for_load()
    :param edition: dict from web.ctx.site.get_many(edition_ids)
    """
    needed_book_fields = []  # book fields that should be copied to the edition
    for book_field, edition_field in AZ_OL_MAP.items():
        if field_value := book.get(book_field) and not edition.get(edition_field):
            needed_book_fields.append(book_field)

    if needed_book_fields == ["authors"]:  # If the only field needed is authors
        if work_key := edition.get("works") and edition["work"][0].get("key"):
            work = web.ctx.site.get(work_key)
            if work.get("authors"):
                needed_book_fields = []

    if needed_book_fields:  # Log book fields that should to be copied to the edition
        fields = ", ".join(needed_book_fields)
        logger.debug(f"{edition.get('key', None) or 'New Edition'} needs {fields}")
    return needed_book_fields


def get_editions_for_books(books: list[dict]) -> list[dict]:
    """
    Get the OL editions for a list of ISBNs.

    :param isbns: list of book dicts
    :return: list of OL editions dicts
    """
    isbns = get_isbns_from_books(books)
    unique_edition_ids = set(
        web.ctx.site.things({'type': '/type/edition', 'isbn_': isbns})
    )
    return web.ctx.site.get_many(list(unique_edition_ids))


def get_pending_books(books):
    pending_books = []
    editions = get_editions_for_books(books)  # Make expensive call just once
    # For each amz book, check that we need its data
    for book in books:
        ed = next(
            (
                ed
                for ed in editions
                if set(book.get('isbn_13')).intersection(set(ed.isbn_13))
                or set(book.get('isbn_10')).intersection(set(ed.isbn_10))
            ),
            {},
        )

        if is_book_needed(book, ed):
            pending_books.append(book)
    return pending_books


def process_amazon_batch(isbn_10s: list[str]) -> None:
    """
    Call the Amazon API to get the products for a list of isbn_10s and store
    each product in memcache using amazon_product_{isbn_13} as the cache key.
    """
    logger.info(f"process_amazon_batch(): {len(isbn_10s)} items")
    try:
        products = web.amazon_api.get_products(isbn_10s, serialize=True)
        # stats_ol_affiliate_amazon_imports - Open Library - Dashboards - Grafana
        # http://graphite.us.archive.org Metrics.stats.ol...
        stats.increment(
            "ol.affiliate.amazon.total_items_fetched",
            n=len(products),
        )
    except Exception:
        logger.exception(f"amazon_api.get_products({isbn_10s}, serialize=True)")
        return

    for product in products:
        cache_key = (  # Make sure we have an isbn_13 for cache key
            product.get('isbn_13')
            and product.get('isbn_13')[0]
            or isbn_10_to_isbn_13(product.get('isbn_10')[0])
        )
        cache.memcache_cache.set(  # Add each product to memcache
            f'amazon_product_{cache_key}', product, expires=WEEK_SECS
        )

    # Only proceed if config finds infobase db creds
    if not config.infobase.get('db_parameters'):  # type: ignore[attr-defined]
        logger.debug("DB parameters missing from affiliate-server infobase")
        return

    if books := [clean_amazon_metadata_for_load(product) for product in products]:
        if pending_books := get_pending_books(books):
            stats.increment(
                "ol.affiliate.amazon.total_items_batched_for_import",
                n=len(pending_books),
            )
            get_current_amazon_batch().add_items(
                [{'ia_id': b['source_records'][0], 'data': b} for b in pending_books]
            )


def seconds_remaining(start_time: float) -> float:
    return max(API_MAX_WAIT_SECONDS - (time.time() - start_time), 0)


def amazon_lookup(site, stats_client, logger) -> None:
    """
    A separate thread of execution that uses the time up to API_MAX_WAIT_SECONDS to
    create a list of isbn_10s that is not larger than API_MAX_ITEMS_PER_CALL and then
    passes them to process_amazon_batch()
    """
    stats.client = stats_client
    web.ctx.site = site

    while True:
        start_time = time.time()
        isbn_10s: set[str] = set()  # no duplicates in the batch
        while len(isbn_10s) < API_MAX_ITEMS_PER_CALL and seconds_remaining(start_time):
            try:  # queue.get() will block (sleep) until successful or it times out
                isbn_10s.add(
                    web.amazon_queue.get(timeout=seconds_remaining(start_time))
                )
            except queue.Empty:
                pass
        logger.info(f"Before amazon_lookup(): {len(isbn_10s)} items")
        if isbn_10s:
            time.sleep(seconds_remaining(start_time))
            try:
                process_amazon_batch(list(isbn_10s))
                logger.info(f"After amazon_lookup(): {len(isbn_10s)} items")
            except Exception:
                logger.exception("Amazon Lookup Thread died")
                stats_client.incr("ol.affiliate.amazon.lookup_thread_died")


def make_amazon_lookup_thread() -> threading.Thread:
    """Called from start_server() and assigned to web.amazon_lookup_thread."""
    thread = threading.Thread(
        target=amazon_lookup,
        args=(web.ctx.site, stats.client, logger),
        daemon=True,
    )
    thread.start()
    return thread


class Status:
    def GET(self) -> str:
        return json.dumps(
            {
                "thread_is_alive": bool(
                    web.amazon_lookup_thread and web.amazon_lookup_thread.is_alive()
                ),
                "queue_size": web.amazon_queue.qsize(),
                "queue": list(web.amazon_queue.queue),
            }
        )


class Clear:
    """Clear web.amazon_queue and return the queue size before it was cleared."""

    def GET(self) -> str:
        qsize = web.amazon_queue.qsize()
        web.amazon_queue.queue.clear()
        stats.put(
            "ol.affiliate.amazon.currently_queued_isbns",
            web.amazon_queue.qsize(),
        )
        return json.dumps({"Cleared": "True", "qsize": qsize})


class Submit:
    @classmethod
    def unpack_isbn(cls, isbn) -> tuple[str, str]:
        isbn = normalize_isbn(isbn.replace('-', ''))
        isbn10 = (
            isbn
            if len(isbn) == 10
            else isbn.startswith('978') and isbn_13_to_isbn_10(isbn)
        )
        isbn13 = isbn if len(isbn) == 13 else isbn_10_to_isbn_13(isbn)
        return isbn10, isbn13

    def GET(self, isbn: str) -> str:
        """
        If isbn is in memcache then return the `hit`.  If not then queue the isbn to be
        looked up and return the equivalent of a promise as `submitted`
        """
        # cache could be None if reached before initialized (mypy)
        if not web.amazon_api:
            return json.dumps({"error": "not_configured"})

        isbn10, isbn13 = self.unpack_isbn(isbn)
        if not isbn10:
            return json.dumps(
                {"error": "rejected_isbn", "isbn10": isbn10, "isbn13": isbn13}
            )

        # Cache lookup by isbn13. If there's a hit return the product to the caller
        if product := cache.memcache_cache.get(f'amazon_product_{isbn13}'):
            return json.dumps({"status": "success", "hit": product})

        # Cache misses will be submitted to Amazon as ASINs (isbn10)
        if isbn10 not in web.amazon_queue.queue:
            web.amazon_queue.put_nowait(isbn10)

        # Give us a snapshot over time of how many new isbns are currently queued
        stats.put(
            "ol.affiliate.amazon.currently_queued_isbns",
            web.amazon_queue.qsize(),
            rate=0.2,
        )
        return json.dumps({"status": "submitted", "queue": web.amazon_queue.qsize()})


def load_config(configfile):
    # This loads openlibrary.yml + infobase.yml
    openlibrary_load_config(configfile)

    stats.client = stats.create_stats_client(cfg=config)

    web.amazon_api = None
    args = [
        config.amazon_api.get('key'),
        config.amazon_api.get('secret'),
        config.amazon_api.get('id'),
    ]
    if all(args):
        web.amazon_api = AmazonAPI(*args, throttling=0.9)
        logger.info("AmazonAPI Initialized")
    else:
        raise RuntimeError(f"{configfile} is missing required keys.")


def setup_env():
    # make sure PYTHON_EGG_CACHE is writable
    os.environ['PYTHON_EGG_CACHE'] = "/tmp/.python-eggs"

    # required when run as fastcgi
    os.environ['REAL_SCRIPT_NAME'] = ""


def start_server():
    sysargs = sys.argv[1:]
    configfile, args = sysargs[0], sysargs[1:]
    web.ol_configfile = configfile

    # # type: (str) -> None

    load_config(web.ol_configfile)

    # sentry loaded by infogami
    infogami._setup()

    if "pytest" not in sys.modules:
        web.amazon_lookup_thread = make_amazon_lookup_thread()
        thread_is_alive = bool(
            web.amazon_lookup_thread and web.amazon_lookup_thread.is_alive()
        )
        logger.critical(f"web.amazon_lookup_thread.is_alive() is {thread_is_alive}")
    else:
        logger.critical("Not starting amazon_lookup_thread in pytest")

    sys.argv = [sys.argv[0]] + list(args)
    app.run()


def start_gunicorn_server():
    """Starts the affiliate server using gunicorn server."""
    from gunicorn.app.base import Application

    configfile = sys.argv.pop(1)

    class WSGIServer(Application):
        def init(self, parser, opts, args):
            pass

        def load(self):
            load_config(configfile)
            # init_setry(app)
            return app.wsgifunc(https_middleware)

    WSGIServer("%prog openlibrary.yml --gunicorn [options]").run()


def https_middleware(app):
    """Hack to support https even when the app server http only.

    The nginx configuration has changed to add the following setting:

        proxy_set_header X-Scheme $scheme;

    Using that value to overwrite wsgi.url_scheme in the WSGI environ,
    which is used by all redirects and other utilities.
    """

    def wrapper(environ, start_response):
        if environ.get('HTTP_X_SCHEME') == 'https':
            environ['wsgi.url_scheme'] = 'https'
        return app(environ, start_response)

    return wrapper


def runfcgi(func, addr=('localhost', 8000)):
    """Runs a WSGI function as a FastCGI pre-fork server."""
    config = dict(web.config.get("fastcgi", {}))

    mode = config.pop("mode", None)
    if mode == "prefork":
        import flup.server.fcgi_fork as flups
    else:
        import flup.server.fcgi as flups

    return flups.WSGIServer(func, multiplexed=True, bindAddress=addr, **config).run()


web.config.debug = False
web.wsgi.runfcgi = runfcgi

app = web.application(urls, locals())


if __name__ == "__main__":
    setup_env()
    if "--gunicorn" in sys.argv:
        sys.argv.pop(sys.argv.index("--gunicorn"))
        start_gunicorn_server()
    else:
        start_server()
