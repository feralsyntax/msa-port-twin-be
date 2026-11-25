from django.test import SimpleTestCase

from port_twin_app.services.terminal_edi.parser import EDIFACTParser


class EDIFACTParserTests(SimpleTestCase):
    def setUp(self):
        self.parser = EDIFACTParser()

    def test_parses_segments_and_composite_elements(self):
        result = self.parser.parse(
            "UNH+1+CODECO:D:95B:UN'BGM+36+MESSAGE-001+9'NAD+CA+TERMINAL-01'UNT+3+1'"
        )

        self.assertEqual(
            [segment.tag for segment in result],
            ["UNH", "BGM", "NAD", "UNT"],
        )
        self.assertEqual(
            result[0].elements[0],
            ("1",),
        )
        self.assertEqual(
            result[0].elements[1],
            ("CODECO", "D", "95B", "UN"),
        )

    def test_parses_custom_una_delimiters(self):
        result = self.parser.parse("UNA;*.,!~UNH*1*CODECO;D;95B;UN~UNT*2*1~")

        self.assertEqual(result[0].tag, "UNH")
        self.assertEqual(
            result[0].elements[1],
            ("CODECO", "D", "95B", "UN"),
        )

    def test_preserves_escaped_element_separator(self):
        result = self.parser.parse("FTX+AAI+TEXT?+VALUE'")

        self.assertEqual(
            result[0].elements[1],
            ("TEXT+VALUE",),
        )

    def test_rejects_empty_message(self):
        with self.assertRaises(ValueError):
            self.parser.parse("   ")

    def test_rejects_incomplete_release_character(self):
        with self.assertRaises(ValueError):
            self.parser.parse("BGM+MESSAGE?")

    def test_rejects_invalid_una(self):
        with self.assertRaises(ValueError):
            self.parser.parse("UNA")
