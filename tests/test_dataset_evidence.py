import json
import tempfile
import unittest
from pathlib import Path

from etm_flow.data.dataset import (
    DatasetBuildConfig,
    FlowSequence,
    build_dataset,
    iter_segment_edges,
    make_node,
)


def evidence_segment():
    return {
        "segment_id": 0,
        "start": {"module": "main", "function": "victim", "function_offset": "0x0"},
        "tokens": [
            {
                "entry": {"module": "main", "function": "victim", "function_offset": "0x0"},
                "src_ctrl": {"module": "main", "function": "victim", "function_offset": "0x4"},
                "ctrl": {"kind": "return", "atom": "E"},
                "dst": {"module": "<UNKNOWN_MODULE>", "function": "<UNKNOWN_TARGET>", "function_offset": "0x0"},
                "instruction_count": 2,
                "quality": {
                    "status": "target_unresolved",
                    "confidence": "unresolved",
                    "target_valid": False,
                    "continuity": "continuous",
                },
            },
            {
                "entry": {"module": "main", "function": "<GAP>", "function_offset": "0x0"},
                "src_ctrl": {"module": "main", "function": "<GAP>", "function_offset": "0x0"},
                "ctrl": {"kind": "flow_gap", "atom": "missing_address"},
                "dst": {"module": "main", "function": "<GAP>", "function_offset": "0x0"},
                "instruction_count": "<unknown>",
                "quality": {
                    "status": "gap",
                    "target_valid": False,
                    "continuity": "continuous",
                },
            },
            {
                "entry": {"module": "libc", "function": "gadget", "function_offset": "0x0"},
                "src_ctrl": {"module": "libc", "function": "gadget", "function_offset": "0x4"},
                "ctrl": {"kind": "return", "atom": "E"},
                "dst": {"module": "main", "function": "next", "function_offset": "0x0"},
                "instruction_count": 2,
                "quality": {
                    "status": "resolved",
                    "confidence": "exact",
                    "target_valid": True,
                    "continuity": "after_gap",
                },
            },
        ],
    }


class EvidenceDatasetTest(unittest.TestCase):
    def test_joint_target_node_is_module_qualified(self) -> None:
        self.assertEqual(make_node("liba", "foo", "0x10"), "liba::foo@0x10")
        self.assertEqual(make_node("libb", "foo", "0x10"), "libb::foo@0x10")
        self.assertNotEqual(
            make_node("liba", "foo", "0x10"),
            make_node("libb", "foo", "0x10"),
        )

    def test_explicit_entry_and_quality_are_encoded(self) -> None:
        edges = list(iter_segment_edges(evidence_segment()))

        self.assertEqual(edges[0]["dst_node"], "<UNKNOWN_TARGET>@0x0")
        self.assertEqual(edges[0]["ctrl_type"], "return:taken|unresolved")
        self.assertEqual(edges[1]["is_gap"], 1)
        self.assertEqual(edges[2]["src_module"], "libc")
        self.assertEqual(edges[2]["dst_module"], "main")
        self.assertEqual(edges[2]["dst_func"], "next")
        self.assertEqual(edges[2]["dst_node"], "main::next@0x0")
        self.assertEqual(edges[2]["transition_valid"], 0)
        self.assertEqual(edges[2]["control_valid"], 1)

    def test_flow_sequence_masks_unknown_targets_and_post_gap_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.jsonl"
            dataset_dir = root / "dataset"
            input_path.write_text(json.dumps(evidence_segment()) + "\n", encoding="utf-8")

            meta = build_dataset(
                [input_path],
                dataset_dir,
                DatasetBuildConfig(min_segment_length=1),
            )
            sequence = FlowSequence(
                dataset_dir,
                max_seq_len=8,
                batch_size=8,
                mode="train",
                validation_split=0.0,
                shuffle=False,
            )
            try:
                inputs, _, weights = sequence[0]

                self.assertEqual(meta["format_version"], 4)
                self.assertIn("src_module", inputs)
                self.assertIn("dst_func", inputs)
                self.assertNotIn("entry_func", inputs)
                self.assertNotIn("dst_node", inputs)
                self.assertEqual(weights["next_dst_node"].tolist(), [0.0, 0.0])
                self.assertEqual(weights["next_ctrl_type"].tolist(), [1.0, 0.0])
            finally:
                # Windows cannot delete an open memmap.  Explicitly release it
                # before TemporaryDirectory cleanup.
                for array in (*sequence.fields.values(), *sequence.aux.values()):
                    array._mmap.close()


if __name__ == "__main__":
    unittest.main()
