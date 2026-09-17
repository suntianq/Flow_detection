import io
import json
import unittest
from pathlib import Path

from etm_flow.data.preprocess import FlowPreprocessor, clean_context, parse_int, split_symbol


class PreprocessHelpersTest(unittest.TestCase):
    def test_parse_int_accepts_decimal_and_hex(self) -> None:
        self.assertEqual(parse_int("42"), 42)
        self.assertEqual(parse_int("0x2a"), 42)
        self.assertIsNone(parse_int(""))

    def test_split_symbol_extracts_offset(self) -> None:
        self.assertEqual(split_symbol("handle_irq+0x10"), ("handle_irq", 16, "+"))

    def test_clean_context_handles_missing_record(self) -> None:
        self.assertEqual(clean_context(None), {})


class EvidenceRecoveryTest(unittest.TestCase):
    @staticmethod
    def process(records, recovery_policy="evidence"):
        stream = io.StringIO()
        preprocessor = FlowPreprocessor(
            sequence_stream=stream,
            include_module_in_token=False,
            target_modules=None,
            min_segment_edges=1,
            recovery_policy=recovery_policy,
        )
        for event_index, record in enumerate(records):
            preprocessor.process_record(Path("sample.jsonl"), event_index, record)
        preprocessor.break_segment("file_end", len(records))
        return [json.loads(line) for line in stream.getvalue().splitlines()]

    def test_unresolved_return_and_gap_are_preserved(self) -> None:
        records = [
            {
                "event": "instruction_range",
                "atom": "E",
                "status": "target_unresolved",
                "reason": "return-target-awaiting-address-packet",
                "runtime_start": "0x1000",
                "runtime_end": "0x1004",
                "elf_start": "0x0",
                "elf_end": "0x4",
                "instruction_count": 2,
                "start_symbol": "victim+0x0",
                "end_symbol": "victim+0x4",
                "waypoint": {"kind": "return", "taken": True},
            },
            {
                "event": "flow_gap",
                "reason": "atom-before-address-target",
                "pc": "0x1004",
            },
            {
                "event": "instruction_range",
                "atom": "E",
                "runtime_start": "0x2000",
                "runtime_end": "0x2004",
                "elf_start": "0x1000",
                "elf_end": "0x1004",
                "instruction_count": 2,
                "start_symbol": "gadget+0x0",
                "end_symbol": "gadget+0x4",
                "waypoint": {"kind": "direct_branch", "taken": True},
                "next_pc": "0x3000",
                "target_source": "static",
                "confidence": "exact",
            },
        ]

        segments = self.process(records)
        self.assertEqual(len(segments), 1)
        tokens = segments[0]["tokens"]
        self.assertEqual([token["ctrl"]["kind"] for token in tokens], [
            "return",
            "flow_gap",
            "direct_branch",
        ])
        self.assertFalse(tokens[0]["quality"]["target_valid"])
        self.assertEqual(tokens[0]["dst"]["function"], "<UNKNOWN_TARGET>")
        self.assertIn("atom-before-address-target", tokens[1]["quality"]["reason"])
        self.assertEqual(tokens[2]["quality"]["continuity"], "after_gap")
        self.assertEqual(tokens[2]["entry"]["function"], "gadget")

    def test_return_stack_mismatch_keeps_expected_and_observed_pc(self) -> None:
        records = [
            {
                "event": "instruction_range",
                "atom": "E",
                "runtime_start": "0x1000",
                "runtime_end": "0x1004",
                "elf_start": "0x0",
                "elf_end": "0x4",
                "start_symbol": "victim+0x0",
                "end_symbol": "victim+0x4",
                "waypoint": {"kind": "return", "taken": True},
                "next_pc": "0x1100",
                "target_source": "software_return_stack",
                "confidence": "speculative",
            },
            {
                "event": "instruction_range",
                "atom": "E",
                "runtime_start": "0x2000",
                "runtime_end": "0x2004",
                "elf_start": "0x1000",
                "elf_end": "0x1004",
                "start_symbol": "gadget+0x0",
                "end_symbol": "gadget+0x4",
                "waypoint": {"kind": "direct_branch", "taken": True},
                "next_pc": "0x3000",
            },
        ]

        tokens = self.process(records)[0]["tokens"]
        self.assertEqual(tokens[0]["quality"]["confidence"], "conflict")
        self.assertFalse(tokens[0]["quality"]["return_stack_match"])
        self.assertEqual(tokens[1]["quality"]["expected_pc"], "0x0000000000001100")
        self.assertEqual(tokens[1]["quality"]["observed_pc"], "0x0000000000002000")

    def test_speculative_path_is_preserved_but_not_treated_as_ground_truth(self) -> None:
        records = [
            {
                "event": "instruction_range",
                "atom": "E",
                "runtime_start": "0x1000",
                "runtime_end": "0x1004",
                "elf_start": "0x0",
                "elf_end": "0x4",
                "start_symbol": "victim+0x0",
                "end_symbol": "victim+0x4",
                "waypoint": {"kind": "return", "taken": True},
                "next_pc": "0x2000",
                "target_source": "software_return_stack",
                "confidence": "speculative",
                "path_confidence": "exact",
            },
            {
                "event": "instruction_range",
                "atom": "E",
                "runtime_start": "0x2000",
                "runtime_end": "0x2004",
                "elf_start": "0x1000",
                "elf_end": "0x1004",
                "start_symbol": "predicted+0x0",
                "end_symbol": "predicted+0x4",
                "waypoint": {"kind": "direct_branch", "taken": True},
                "next_pc": "0x2100",
                "target_source": "static",
                "confidence": "exact",
                "path_confidence": "speculative",
            },
        ]

        tokens = self.process(records)[0]["tokens"]
        self.assertFalse(tokens[0]["quality"]["target_valid"])
        self.assertTrue(tokens[0]["quality"]["control_valid"])
        self.assertEqual(tokens[0]["quality"]["confidence"], "speculative")
        self.assertFalse(tokens[1]["quality"]["target_valid"])
        self.assertFalse(tokens[1]["quality"]["control_valid"])
        self.assertEqual(tokens[1]["quality"]["path_confidence"], "speculative")

    def test_address_packet_confirms_software_return_target(self) -> None:
        records = [
            {
                "event": "instruction_range",
                "atom": "E",
                "runtime_start": "0x1000",
                "runtime_end": "0x1004",
                "elf_start": "0x0",
                "elf_end": "0x4",
                "start_symbol": "victim+0x0",
                "end_symbol": "victim+0x4",
                "waypoint": {"kind": "return", "taken": True},
                "next_pc": "0x2000",
                "target_source": "software_return_stack",
                "confidence": "speculative",
            },
            {
                "event": "address",
                "runtime_addr": "0x2000",
                "elf_addr": "0x1000",
                "symbol": "caller+0x0",
            },
            {
                "event": "instruction_range",
                "atom": "E",
                "runtime_start": "0x2000",
                "runtime_end": "0x2004",
                "elf_start": "0x1000",
                "elf_end": "0x1004",
                "start_symbol": "caller+0x0",
                "end_symbol": "caller+0x4",
                "waypoint": {"kind": "direct_branch", "taken": True},
                "next_pc": "0x3000",
            },
        ]

        first = self.process(records)[0]["tokens"][0]
        self.assertTrue(first["quality"]["target_valid"])
        self.assertTrue(first["quality"]["return_stack_match"])
        self.assertEqual(first["quality"]["confidence"], "confirmed")

    def test_cross_module_stripped_target_remains_continuous(self) -> None:
        records = [
            {
                "event": "instruction_range",
                "module": "main",
                "atom": "E",
                "runtime_start": "0x1000",
                "runtime_end": "0x1004",
                "elf_start": "0x100",
                "elf_end": "0x104",
                "start_symbol": "caller+0x0",
                "end_symbol": "caller+0x4",
                "waypoint": {"kind": "direct_branch", "taken": True},
                "next_pc": "0x2000",
            },
            {
                "event": "instruction_range",
                "module": "libc",
                "atom": "E",
                "runtime_start": "0x2000",
                "runtime_end": "0x2004",
                "elf_start": "0x500",
                "elf_end": "0x504",
                "waypoint": {"kind": "return", "taken": True},
                "next_pc": "0x3000",
                "target_source": "address_packet",
                "confidence": "exact",
            },
        ]

        segments = self.process(records)
        self.assertEqual(len(segments), 1)
        tokens = segments[0]["tokens"]
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0]["src_ctrl"]["module"], "main")
        self.assertEqual(tokens[0]["dst"]["module"], "libc")
        self.assertEqual(tokens[0]["dst"]["function"], "<elf>")
        self.assertEqual(tokens[0]["dst"]["function_offset"], "0x500")
        self.assertEqual(tokens[1]["src_ctrl"]["module"], "libc")

    def test_unconfigured_executable_map_keeps_opaque_identity(self) -> None:
        records = [
            {
                "event": "address",
                "address": "0x70001234",
                "reason": "no-configured-elf",
                "map": "/system/lib64/libvendor.so",
                "map_file_offset": "0x2234",
            }
        ]

        tokens = self.process(records)[0]["tokens"]
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["dst"]["module"], "libvendor.so")
        self.assertEqual(tokens[0]["dst"]["function"], "<opaque>")
        self.assertEqual(tokens[0]["dst"]["function_offset"], "0x2234")
        self.assertEqual(tokens[0]["quality"]["status"], "gap")

    def test_strict_policy_keeps_legacy_filtering(self) -> None:
        records = [
            {
                "event": "instruction_range",
                "atom": "E",
                "status": "target_unresolved",
                "runtime_start": "0x1000",
                "runtime_end": "0x1004",
                "elf_start": "0x0",
                "elf_end": "0x4",
                "start_symbol": "victim+0x0",
                "end_symbol": "victim+0x4",
                "waypoint": {"kind": "return", "taken": True},
            }
        ]

        self.assertEqual(self.process(records, recovery_policy="strict"), [])


if __name__ == "__main__":
    unittest.main()
