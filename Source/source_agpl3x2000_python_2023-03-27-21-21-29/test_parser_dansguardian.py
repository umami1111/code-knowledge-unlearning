# SPDX-FileCopyrightText: 2015 Sebastian Wagner
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
import os
import unittest

import intelmq.lib.test as test
import intelmq.lib.utils as utils
from intelmq.bots.parsers.malwarepatrol.parser_dansguardian import \
    DansParserBot


with open(os.path.join(os.path.dirname(__file__), 'getfile')) as handle:
    EXAMPLE_FILE = handle.read()


REPORT = {'__type': 'Report',
          'raw': utils.base64_encode(EXAMPLE_FILE),
          'feed.url': 'https://lists.malwarepatrol.net/cgi/getfile?receipt=wehaveno1dea-howthiskeylookslike!&product=8&list=dansguardian'}
EVENT1 = {'__type': 'Event',
          'raw': 'IwojICAgICAgICBNYWx3YXJlIFBhdHJvbCAtIEJsb2NrIExpc3QgLSBodHRwczovL3d3dy5tYWx3YXJlcGF0cm9sLm5ldAojCUxpc3QgZm9yIERhbnNHdWFyZGlhbgojCUdlbmVyYXRlZCBhdDogMjAxNzAyMjYyMzUzNTYgVVRDCiMKIwlQbGVhc2UgZG8gbm90IHVwZGF0ZSB0aGlzIGxpc3QgbW9yZSBvZnRlbiB0aGFuIGV2ZXJ5IGhvdXIuCiMKIwlDb3B5cmlnaHQgKGMpICAyMDE3IC0gQW5kcmUgQ29ycmVhIC0gTWFsd2FyZSBQYXRyb2wgLSBNYWx3YXJlIEJsb2NrIExpc3QKIwlUaGlzIGluZm9ybWF0aW9uIGlzIHByb3ZpZGVkIGFzLWlzIGFuZCB1bmRlciB0aGUgVGVybXMgYW5kIENvbmRpdGlvbnMKIwlhdmFpbGFibGUgaW4gdGhlIGZvbGxvd2luZyBhZGRyZXNzOgojCiMJaHR0cHM6Ly93d3cubWFsd2FyZXBhdHJvbC5uZXQvdGVybXMuc2h0bWwKIwojCVVzaW5nIHRoaXMgaW5mb3JtYXRpb24gaW5kaWNhdGVzIHlvdXIgYWdyZWVtZW50IHRvIGJlIGJvdW5kIGJ5IHRoZXNlCiMJdGVybXMuIElmIHlvdSBkbyBub3QgYWNjZXB0IHRoZW0sIHBsZWFzZSBkZWxldGUgdGhpcyBmaWxlIGltbWVkaWF0ZWx5LgojCiMJWW91IGNhbiByZXBvcnQgZmFsc2UgcG9zaXRpdmVzIG9yIGJyb2tlbiBydWxlcy9zaWduYXR1cmVzIHRvOgojCWZwIChhIHQpIG1hbHdhcmVwYXRyb2wubmV0CiMKZXhhbXBsZS5jb20vbWFsd2FyZWhlcmUv',
          'feed.url': 'https://lists.malwarepatrol.net/cgi/getfile?&product=8&list=dansguardian',
          'source.url': 'http://example.com/malwarehere/',
          'time.source': '2017-02-26T23:53:56+00:00',
          'classification.type': 'malware-distribution',
          }
EVENT2 = {'__type': 'Event',
          'raw': 'IwojICAgICAgICBNYWx3YXJlIFBhdHJvbCAtIEJsb2NrIExpc3QgLSBodHRwczovL3d3dy5tYWx3YXJlcGF0cm9sLm5ldAojCUxpc3QgZm9yIERhbnNHdWFyZGlhbgojCUdlbmVyYXRlZCBhdDogMjAxNzAyMjYyMzUzNTYgVVRDCiMKIwlQbGVhc2UgZG8gbm90IHVwZGF0ZSB0aGlzIGxpc3QgbW9yZSBvZnRlbiB0aGFuIGV2ZXJ5IGhvdXIuCiMKIwlDb3B5cmlnaHQgKGMpICAyMDE3IC0gQW5kcmUgQ29ycmVhIC0gTWFsd2FyZSBQYXRyb2wgLSBNYWx3YXJlIEJsb2NrIExpc3QKIwlUaGlzIGluZm9ybWF0aW9uIGlzIHByb3ZpZGVkIGFzLWlzIGFuZCB1bmRlciB0aGUgVGVybXMgYW5kIENvbmRpdGlvbnMKIwlhdmFpbGFibGUgaW4gdGhlIGZvbGxvd2luZyBhZGRyZXNzOgojCiMJaHR0cHM6Ly93d3cubWFsd2FyZXBhdHJvbC5uZXQvdGVybXMuc2h0bWwKIwojCVVzaW5nIHRoaXMgaW5mb3JtYXRpb24gaW5kaWNhdGVzIHlvdXIgYWdyZWVtZW50IHRvIGJlIGJvdW5kIGJ5IHRoZXNlCiMJdGVybXMuIElmIHlvdSBkbyBub3QgYWNjZXB0IHRoZW0sIHBsZWFzZSBkZWxldGUgdGhpcyBmaWxlIGltbWVkaWF0ZWx5LgojCiMJWW91IGNhbiByZXBvcnQgZmFsc2UgcG9zaXRpdmVzIG9yIGJyb2tlbiBydWxlcy9zaWduYXR1cmVzIHRvOgojCWZwIChhIHQpIG1hbHdhcmVwYXRyb2wubmV0CiMKd3d3LmV4YW1wbGUubmV0Lw==',
          'feed.url': 'https://lists.malwarepatrol.net/cgi/getfile?&product=8&list=dansguardian',
          'source.url': 'http://www.example.net/',
          'time.source': '2017-02-26T23:53:56+00:00',
          'classification.type': 'malware-distribution',
          }


class TestDansParserBot(test.BotTestCase, unittest.TestCase):
    """
    A TestCase for DansParserBot.
    """

    @classmethod
    def set_bot(cls):
        cls.bot_reference = DansParserBot
        cls.default_input_message = REPORT

    def test_empty(self):
        self.run_bot()
        self.assertMessageEqual(0, EVENT1)
        self.assertMessageEqual(1, EVENT2)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()