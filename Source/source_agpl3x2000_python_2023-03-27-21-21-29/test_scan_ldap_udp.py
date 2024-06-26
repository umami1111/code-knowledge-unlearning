# SPDX-FileCopyrightText: 2019 Guillermo Rodriguez
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-

import os
import unittest

import intelmq.lib.test as test
import intelmq.lib.utils as utils
from intelmq.bots.parsers.shadowserver.parser import ShadowserverParserBot

with open(os.path.join(os.path.dirname(__file__), 'testdata/scan_ldap_udp.csv')) as handle:
    EXAMPLE_FILE = handle.read()
EXAMPLE_LINES = EXAMPLE_FILE.splitlines()

EXAMPLE_REPORT = {'feed.name': 'Open LDAP',
                  "raw": utils.base64_encode(EXAMPLE_FILE),
                  "__type": "Report",
                  "time.observation": "2015-01-01T00:00:00+00:00",
                  "extra.file_name": "2019-01-01-scan_ldap_udp-test-geo.csv",
                  }
EVENTS = [
{
    '__type': 'Event',
    'classification.identifier': 'open-ldap',
    'classification.taxonomy': 'vulnerable',
    'classification.type': 'vulnerable-system',
    'extra.amplification': 58.42,
    'extra.configuration_naming_context': 'CN=Configuration,DC=ad,DC=example,DC=com',
    'extra.current_time': '20220821044533.0Z',
    'extra.default_naming_context': 'DC=ad,DC=example,DC=com',
    'extra.domain_controller_functionality': 7,
    'extra.domain_functionality': 7,
    'extra.ds_service_name': 'CN=Configuration,DC=ad,DC=example,DC=com',
    'extra.forest_functionality': 7,
    'extra.highest_committed_usn': 222537,
    'extra.is_global_catalog_ready': True,
    'extra.is_synchronized': True,
    'extra.ldap_service_name': 'node01.example.com',
    'extra.naming_contexts': 'DC=ad,DC=example,DC=com|CN=Configuration,DC=example,DC=com|CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.root_domain_naming_context': 'DC=example,DC=com',
    'extra.schema_naming_context': 'CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.server_name': 'CN=Servers,CN=Default-First-Site-Name,CN=Sites,CN=Configuration,DC=example,DC=com',
    'extra.size': 3038,
    'extra.subschema_subentry': 'CN=Aggregate,CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.supported_capabilities': '1.2.840.113556.1.4.800|1.2.840.113556.1.4.1670|1.2.840.113556.1.4.1791|1.2.840.113556.1.4.1935|1.2.840.113556.1.4.2080|1.2.840.113556.1.4.2237',
    'extra.supported_control': '1.2.840.113556.1.4.319|1.2.840.113556.1.4.801|1.2.840.113556.1.4.473|1.2.840.113556.1.4.528|1.2.840.113556.1.4.417|1.2.840.113556.1.4.619|1.2.840.113556.1.4.841|1.2.840.113556.1.4.529|1.2.840.113556.1.4.805|1.2.840.113556.1.4.521|1.2.840.113556.1.4.970|1.2.840.113556.1.4.1338|1.2.840.113556.1.4.474|1.2.840.113556.1.4.1339|1.2.840.113556.1.4.1340|1.2.840.113556.1.4.1413|2.16.840.1.113730.3.4.9|2.16.840.1.113730.3.4.10|1.2.840.113556.1.4.1504|1.2.840.113556.1.4.1852|1.2.840.113556.1.4.802|1.2.840.113556.1.4.1907|1.2.840.113556.1.4.1948|1.2.840.113556.1.4.1974|1.2.840.113556.1.4.1341|1.2.840.113556.1.4.2026|1.2.840.113556.1.4.2064|1.2.840.113556.1.4.2065|1.2.840.113556.1.4.2066|1.2.840.113556.1.4.2090|1.2.840.113556.1.4.2205|1.2.840.113556.1.4.2204|1.2.840.113556.1.4.2206|1.2.840.113556.1.4.2211|1.2.840.113556.1.4.2239|1.2.840.113556.1.4.2255|1.2.840.113556.1.4.2256|1.2.840.113556.1.4.2309|1.2.840.113556.1.4.2330|1.2.840.113556.1.4.2354',
    'extra.supported_ldap_policies': 'MaxPoolThreads|MaxPercentDirSyncRequests|MaxDatagramRecv|MaxReceiveBuffer|InitRecvTimeout|MaxConnections|MaxConnIdleTime|MaxPageSize|MaxBatchReturnMessages|MaxQueryDuration|MaxDirSyncDuration|MaxTempTableSize|MaxResultSetSize|MinResultSets|MaxResultSetsPerConn|MaxNotificationPerConn|MaxValRange|MaxValRangeTransitive|ThreadMemoryLimit|SystemMemoryLimitPercent',
    'extra.supported_ldap_version': '3|2',
    'extra.supported_sasl_mechanisms': 'GSSAPI|GSS-SPNEGO|EXTERNAL|DIGEST-MD5',
    'extra.tag': 'ldap-udp',
    'feed.name': 'Open LDAP',
    'protocol.application': 'ldap',
    'protocol.transport': 'udp',
    'raw': utils.base64_encode('\n'.join([EXAMPLE_LINES[0], EXAMPLE_LINES[1]])),
    'source.asn': 64512,
    'source.geolocation.cc': 'ZZ',
    'source.geolocation.city': 'City',
    'source.geolocation.region': 'Region',
    'source.ip': '192.168.0.1',
    'source.local_hostname': 'node01.example.com',
    'source.port': 389,
    'source.reverse_dns': 'node01.example.com',
    'time.source': '2010-02-10T00:00:00+00:00'
},
{
    '__type': 'Event',
    'classification.identifier': 'open-ldap',
    'classification.taxonomy': 'vulnerable',
    'classification.type': 'vulnerable-system',
    'extra.amplification': 58.88,
    'extra.configuration_naming_context': 'CN=Configuration,DC=ad,DC=example,DC=com',
    'extra.current_time': '20220821044948.0Z',
    'extra.default_naming_context': 'DC=ad,DC=example,DC=com',
    'extra.domain_controller_functionality': 7,
    'extra.domain_functionality': 7,
    'extra.ds_service_name': 'CN=Configuration,DC=ad,DC=example,DC=com',
    'extra.forest_functionality': 7,
    'extra.highest_committed_usn': 1478714,
    'extra.is_global_catalog_ready': True,
    'extra.is_synchronized': True,
    'extra.ldap_service_name': 'node02.example.com',
    'extra.naming_contexts': 'DC=ad,DC=example,DC=com|CN=Configuration,DC=example,DC=com|CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.root_domain_naming_context': 'DC=example,DC=com',
    'extra.schema_naming_context': 'CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.server_name': 'CN=Servers,CN=Default-First-Site-Name,CN=Sites,CN=Configuration,DC=example,DC=com',
    'extra.size': 3062,
    'extra.subschema_subentry': 'CN=Aggregate,CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.supported_capabilities': '1.2.840.113556.1.4.800|1.2.840.113556.1.4.1670|1.2.840.113556.1.4.1791|1.2.840.113556.1.4.1935|1.2.840.113556.1.4.2080|1.2.840.113556.1.4.2237',
    'extra.supported_control': '1.2.840.113556.1.4.319|1.2.840.113556.1.4.801|1.2.840.113556.1.4.473|1.2.840.113556.1.4.528|1.2.840.113556.1.4.417|1.2.840.113556.1.4.619|1.2.840.113556.1.4.841|1.2.840.113556.1.4.529|1.2.840.113556.1.4.805|1.2.840.113556.1.4.521|1.2.840.113556.1.4.970|1.2.840.113556.1.4.1338|1.2.840.113556.1.4.474|1.2.840.113556.1.4.1339|1.2.840.113556.1.4.1340|1.2.840.113556.1.4.1413|2.16.840.1.113730.3.4.9|2.16.840.1.113730.3.4.10|1.2.840.113556.1.4.1504|1.2.840.113556.1.4.1852|1.2.840.113556.1.4.802|1.2.840.113556.1.4.1907|1.2.840.113556.1.4.1948|1.2.840.113556.1.4.1974|1.2.840.113556.1.4.1341|1.2.840.113556.1.4.2026|1.2.840.113556.1.4.2064|1.2.840.113556.1.4.2065|1.2.840.113556.1.4.2066|1.2.840.113556.1.4.2090|1.2.840.113556.1.4.2205|1.2.840.113556.1.4.2204|1.2.840.113556.1.4.2206|1.2.840.113556.1.4.2211|1.2.840.113556.1.4.2239|1.2.840.113556.1.4.2255|1.2.840.113556.1.4.2256|1.2.840.113556.1.4.2309|1.2.840.113556.1.4.2330|1.2.840.113556.1.4.2354',
    'extra.supported_ldap_policies': 'MaxPoolThreads|MaxPercentDirSyncRequests|MaxDatagramRecv|MaxReceiveBuffer|InitRecvTimeout|MaxConnections|MaxConnIdleTime|MaxPageSize|MaxBatchReturnMessages|MaxQueryDuration|MaxDirSyncDuration|MaxTempTableSize|MaxResultSetSize|MinResultSets|MaxResultSetsPerConn|MaxNotificationPerConn|MaxValRange|MaxValRangeTransitive|ThreadMemoryLimit|SystemMemoryLimitPercent',
    'extra.supported_ldap_version': '3|2',
    'extra.supported_sasl_mechanisms': 'GSSAPI|GSS-SPNEGO|EXTERNAL|DIGEST-MD5',
    'extra.tag': 'ldap-udp',
    'feed.name': 'Open LDAP',
    'protocol.application': 'ldap',
    'protocol.transport': 'udp',
    'raw': utils.base64_encode('\n'.join([EXAMPLE_LINES[0], EXAMPLE_LINES[2]])),
    'source.asn': 64512,
    'source.geolocation.cc': 'ZZ',
    'source.geolocation.city': 'City',
    'source.geolocation.region': 'Region',
    'source.ip': '192.168.0.2',
    'source.local_hostname': 'node02.example.com',
    'source.port': 389,
    'source.reverse_dns': 'node02.example.com',
    'time.source': '2010-02-10T00:00:01+00:00'
},
{
    '__type': 'Event',
    'classification.identifier': 'open-ldap',
    'classification.taxonomy': 'vulnerable',
    'classification.type': 'vulnerable-system',
    'extra.amplification': 0.69,
    'extra.configuration_naming_context': 'CN=Configuration,DC=ad,DC=example,DC=com',
    'extra.default_naming_context': 'DC=ad,DC=example,DC=com',
    'extra.ds_service_name': 'CN=Configuration,DC=ad,DC=example,DC=com',
    'extra.ldap_service_name': 'node03.example.com',
    'extra.naming_contexts': 'DC=ad,DC=example,DC=com|CN=Configuration,DC=example,DC=com|CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.root_domain_naming_context': 'DC=example,DC=com',
    'extra.schema_naming_context': 'CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.server_name': 'CN=Servers,CN=Default-First-Site-Name,CN=Sites,CN=Configuration,DC=example,DC=com',
    'extra.size': 36,
    'extra.subschema_subentry': 'CN=Aggregate,CN=Schema,CN=Configuration,DC=example,DC=com',
    'extra.tag': 'ldap-udp',
    'feed.name': 'Open LDAP',
    'protocol.application': 'ldap',
    'protocol.transport': 'udp',
    'raw': utils.base64_encode('\n'.join([EXAMPLE_LINES[0], EXAMPLE_LINES[3]])),
    'source.asn': 64512,
    'source.geolocation.cc': 'ZZ',
    'source.geolocation.city': 'City',
    'source.geolocation.region': 'Region',
    'source.ip': '192.168.0.3',
    'source.local_hostname': 'node03.example.com',
    'source.port': 389,
    'source.reverse_dns': 'node03.example.com',
    'time.source': '2010-02-10T00:00:02+00:00'
}
          ]

class TestShadowserverParserBot(test.BotTestCase, unittest.TestCase):
    """
    A TestCase for a ShadowserverParserBot.
    """

    @classmethod
    def set_bot(cls):
        cls.bot_reference = ShadowserverParserBot
        cls.default_input_message = EXAMPLE_REPORT

    def test_event(self):
        """ Test if correct Event has been produced. """
        self.run_bot()
        for i, EVENT in enumerate(EVENTS):
            self.assertMessageEqual(i, EVENT)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
