import os

from openlibrary.catalog.marc.marc_binary import BinaryDataField, MarcBinary

test_data = "%s/test_data/bin_input/" % os.path.dirname(__file__)


class MockMARC:
    def __init__(self, encoding):
        """
        :param encoding str: 'utf8' or 'marc8'
        """
        self.encoding = encoding

    def marc8(self):
        return self.encoding == 'marc8'


def test_wrapped_lines():
    filename = '%s/wrapped_lines.mrc' % test_data
    with open(filename, 'rb') as f:
        rec = MarcBinary(f.read())
        ret = list(rec.read_fields(['520']))
        assert len(ret) == 2
        a, b = ret
        assert a[0] == '520' and b[0] == '520'
        a_content = list(a[1].get_all_subfields())[0][1]
        assert len(a_content) == 2290
        b_content = list(b[1].get_all_subfields())[0][1]
        assert len(b_content) == 243


class Test_BinaryDataField:
    def test_translate(self):
        bdf = BinaryDataField(MockMARC('marc8'), b'')
        assert (
            bdf.translate(b'Vieira, Claudio Bara\xe2una,') == 'Vieira, Claudio Baraúna,'
        )

    def test_bad_marc_line(self):
        line = (
            b'0 \x1f\xe2aEtude objective des ph\xe2enom\xe1enes neuro-psychiques;\x1e'
        )
        bdf = BinaryDataField(MockMARC('marc8'), line)
        assert list(bdf.get_all_subfields()) == [
            ('á', 'Etude objective des phénomènes neuro-psychiques;')
        ]


class Test_MarcBinary:
    def test_all_fields(self):
        filename = '%s/onquietcomedyint00brid_meta.mrc' % test_data
        with open(filename, 'rb') as f:
            rec = MarcBinary(f.read())
            fields = list(rec.all_fields())
            assert len(fields) == 13
            assert fields[0][0] == '001'
            for f, v in fields:
                if f == '001':
                    f001 = v
                elif f == '008':
                    f008 = v
                elif f == '100':
                    f100 = v
            assert isinstance(f001, str)
            assert isinstance(f008, str)
            assert isinstance(f100, BinaryDataField)

    def test_get_subfield_value(self):
        filename = '%s/onquietcomedyint00brid_meta.mrc' % test_data
        with open(filename, 'rb') as f:
            rec = MarcBinary(f.read())
            rec.build_fields(['100', '245', '010'])
            author_field = rec.get_fields('100')
            assert isinstance(author_field, list)
            assert isinstance(author_field[0], BinaryDataField)
            subfields = author_field[0].get_subfields('a')
            assert next(subfields) == ('a', 'Bridgham, Gladys Ruth. [from old catalog]')
            values = author_field[0].get_subfield_values('a')
            (name,) = values  # 100$a is non-repeatable, there will be only one
            assert name == 'Bridgham, Gladys Ruth. [from old catalog]'
