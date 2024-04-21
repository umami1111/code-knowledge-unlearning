"""Interface to import queue.
"""
from collections import defaultdict
from typing import Any

import logging
import datetime
import time
import web
import json

from psycopg2.errors import UndefinedTable, UniqueViolation

from . import db

logger = logging.getLogger("openlibrary.imports")


class Batch(web.storage):
    @staticmethod
    def find(name, create=False):
        result = db.query("SELECT * FROM import_batch where name=$name", vars=locals())
        if result:
            return Batch(result[0])
        elif create:
            return Batch.new(name)

    @staticmethod
    def new(name):
        db.insert("import_batch", name=name)
        return Batch.find(name=name)

    def load_items(self, filename):
        """Adds all the items specified in the filename to this batch."""
        items = [line.strip() for line in open(filename) if line.strip()]
        self.add_items(items)

    def dedupe_items(self, items):
        ia_ids = [item.get('ia_id') for item in items if item.get('ia_id')]
        already_present = {
            row.ia_id
            for row in db.query(
                "SELECT ia_id FROM import_item WHERE ia_id IN $ia_ids",
                vars={"ia_ids": ia_ids},
            )
        }
        # ignore already present
        logger.info(
            "batch %s: %d items are already present, ignoring...",
            self.name,
            len(already_present),
        )
        # Those unique items whose ia_id's aren't already present
        return [item for item in items if item.get('ia_id') not in already_present]

    def normalize_items(self, items):
        return [
            {'ia_id': item}
            if type(item) is str
            else {
                'batch_id': self.id,
                # Partner bots set ia_id to eg "partner:978..."
                'ia_id': item.get('ia_id'),
                'data': json.dumps(item.get('data'), sort_keys=True)
                if item.get('data')
                else None,
            }
            for item in items
        ]

    def add_items(self, items: list[str] | list[dict]) -> None:
        """
        :param items: either a list of `ia_id`  (legacy) or a list of dicts
            containing keys `ia_id` and book `data`. In the case of
            the latter, `ia_id` will be of form e.g. "isbn:1234567890";
            i.e. of a format id_type:value which cannot be a valid IA id.
        """
        if not items:
            return

        logger.info("batch %s: adding %d items", self.name, len(items))

        items = self.dedupe_items(self.normalize_items(items))
        if items:
            try:
                # TODO: Upgrade psql and use `INSERT OR IGNORE`
                # otherwise it will fail on UNIQUE `data`
                # https://stackoverflow.com/questions/1009584
                db.get_db().multiple_insert("import_item", items)
            except UniqueViolation:
                for item in items:
                    try:
                        db.get_db().insert("import_item", **item)
                    except UniqueViolation:
                        pass
            logger.info("batch %s: added %d items", self.name, len(items))

        return

    def get_items(self, status="pending"):
        result = db.where("import_item", batch_id=self.id, status=status)
        return [ImportItem(row) for row in result]


class ImportItem(web.storage):
    @staticmethod
    def find_pending(limit=1000):
        result = db.where("import_item", status="pending", order="id", limit=limit)
        return map(ImportItem, result)

    @staticmethod
    def find_by_identifier(identifier):
        result = db.where("import_item", ia_id=identifier)
        if result:
            return ImportItem(result[0])

    def set_status(self, status, error=None, ol_key=None):
        id_ = self.ia_id or f"{self.batch_id}:{self.id}"
        logger.info("set-status %s - %s %s %s", id_, status, error, ol_key)
        d = dict(
            status=status,
            error=error,
            ol_key=ol_key,
            import_time=datetime.datetime.utcnow(),
        )
        if status != 'failed':
            d = dict(**d, data=None)
        db.update("import_item", where="id=$id", vars=self, **d)
        self.update(d)

    def mark_failed(self, error):
        self.set_status(status='failed', error=error)

    def mark_found(self, ol_key):
        self.set_status(status='found', ol_key=ol_key)

    def mark_created(self, ol_key):
        self.set_status(status='created', ol_key=ol_key)

    def mark_modified(self, ol_key):
        self.set_status(status='modified', ol_key=ol_key)

    @classmethod
    def delete_items(
        cls, ia_ids: list[str], batch_id: int | None = None, _test: bool = False
    ):
        oldb = db.get_db()
        data: dict[str, Any] = {
            'ia_ids': ia_ids,
        }

        where = 'ia_id IN $ia_ids'

        if batch_id:
            data['batch_id'] = batch_id
            where += ' AND batch_id=$batch_id'

        return oldb.delete('import_item', where=where, vars=data, _test=_test)


class Stats:
    """Import Stats."""

    def get_imports_per_hour(self):
        """Returns the number imports happened in past one hour duration."""
        try:
            result = db.query(
                "SELECT count(*) as count FROM import_item"
                + " WHERE import_time > CURRENT_TIMESTAMP - interval '1' hour"
            )
        except UndefinedTable:
            logger.exception("Database table import_item may not exist on localhost")
            return 0
        return result[0].count

    def get_count(self, status=None):
        where = "status=$status" if status else "1=1"
        try:
            rows = db.select(
                "import_item", what="count(*) as count", where=where, vars=locals()
            )
        except UndefinedTable:
            logger.exception("Database table import_item may not exist on localhost")
            return 0
        return rows[0].count

    def get_count_by_status(self, date=None):
        rows = db.query("SELECT status, count(*) FROM import_item GROUP BY status")
        return {row.status: row.count for row in rows}

    def get_count_by_date_status(self, ndays=10):
        try:
            result = db.query(
                "SELECT added_time::date as date, status, count(*)"
                + " FROM import_item "
                + " WHERE added_time > current_date - interval '$ndays' day"
                " GROUP BY 1, 2" + " ORDER BY 1 desc",
                vars=locals(),
            )
        except UndefinedTable:
            logger.exception("Database table import_item may not exist on localhost")
            return []
        d = defaultdict(dict)
        for row in result:
            d[row.date][row.status] = row.count
        return sorted(d.items(), reverse=True)

    def get_books_imported_per_day(self):
        try:
            rows = db.query(
                "SELECT import_time::date as date, count(*) as count"
                " FROM import_item" + " WHERE status='created'"
                " GROUP BY 1" + " ORDER BY 1"
            )
        except UndefinedTable:
            logger.exception("Database table import_item may not exist on localhost")
            return []
        return [[self.date2millis(row.date), row.count] for row in rows]

    def date2millis(self, date):
        return time.mktime(date.timetuple()) * 1000

    def get_items(self, date=None, order=None, limit=None):
        """Returns all rows with given added date."""
        where = "added_time::date = $date" if date else "1 = 1"
        try:
            return db.select(
                "import_item", where=where, order=order, limit=limit, vars=locals()
            )
        except UndefinedTable:
            logger.exception("Database table import_item may not exist on localhost")
            return []

    def get_items_summary(self, date):
        """Returns all rows with given added date."""
        rows = db.query(
            "SELECT status, count(*) as count"
            + " FROM import_item"
            + " WHERE added_time::date = $date"
            " GROUP BY status",
            vars=locals(),
        )
        return {"counts": {row.status: row.count for row in rows}}
