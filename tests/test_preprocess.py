import unittest

from etm_flow.data.preprocess import clean_context, parse_int, split_symbol


class PreprocessHelpersTest(unittest.TestCase):
    def test_parse_int_accepts_decimal_and_hex(self) -> None:
        self.assertEqual(parse_int("42"), 42)
        self.assertEqual(parse_int("0x2a"), 42)
        self.assertIsNone(parse_int(""))

    def test_split_symbol_extracts_offset(self) -> None:
        self.assertEqual(split_symbol("handle_irq+0x10"), ("handle_irq", 16, "+"))

    def test_clean_context_handles_missing_record(self) -> None:
        self.assertEqual(clean_context(None), {})


if __name__ == "__main__":
    unittest.main()
