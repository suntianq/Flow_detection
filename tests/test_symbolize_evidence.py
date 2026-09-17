import io
import json
import unittest
from pathlib import Path

from etm_flow.trace.symbolize import (
    ContextState,
    DecodedInstruction,
    EventWriter,
    MapEntry,
    ModuleRule,
    Resolution,
)


class SymbolizeEvidenceTest(unittest.TestCase):
    def test_instruction_range_writes_target_and_path_provenance(self) -> None:
        stream = io.StringIO()
        writer = EventWriter(stream)
        module = ModuleRule(name="libc", elf_path=Path("libc.so"))
        resolution = Resolution(
            runtime_addr=0x1000,
            mapping=None,
            module=module,
            elf_va=0,
            file_offset=0,
            segment=None,
            section=None,
            symbol=None,
            symbol_offset=None,
            symbol_contains=False,
            source=None,
            reason=None,
        )
        waypoint = DecodedInstruction(
            runtime_addr=0x1000,
            raw=b"\xc0\x03\x5f\xd6",
            word=0xD65F03C0,
            mnemonic="ret",
            op_str="",
            kind="return",
        )

        writer.instruction_range_line(
            ContextState(),
            resolution,
            resolution,
            waypoint,
            instruction_count=1,
            atom="E",
            taken=True,
            next_pc=0x2000,
            target_source="software_return_stack",
            confidence="speculative",
            path_confidence="exact",
        )

        record = json.loads(stream.getvalue())
        self.assertEqual(record["target_source"], "software_return_stack")
        self.assertEqual(record["module"], "libc")
        self.assertEqual(record["confidence"], "speculative")
        self.assertEqual(record["path_confidence"], "exact")

    def test_unconfigured_module_address_keeps_file_relative_offset(self) -> None:
        stream = io.StringIO()
        writer = EventWriter(stream)
        mapping = MapEntry(
            start=0x70000000,
            end=0x70010000,
            perms="r-xp",
            pgoff=0x1000,
            device="00:00",
            inode="0",
            path="/system/lib64/libvendor.so",
            line_no=1,
        )
        resolution = Resolution(
            runtime_addr=0x70000234,
            mapping=mapping,
            module=None,
            elf_va=None,
            file_offset=None,
            segment=None,
            section=None,
            symbol=None,
            symbol_offset=None,
            symbol_contains=False,
            source=None,
            reason="no-configured-elf",
        )

        writer.address_line(ContextState(), resolution)

        record = json.loads(stream.getvalue())
        self.assertEqual(record["map"], "/system/lib64/libvendor.so")
        self.assertEqual(record["map_file_offset"], "0x1234")
        self.assertEqual(record["reason"], "no-configured-elf")


if __name__ == "__main__":
    unittest.main()
