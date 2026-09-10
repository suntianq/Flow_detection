#!/usr/bin/env python3
"""Build and load flow-sequence datasets for next-edge prediction.

Input files are the JSONL files produced by ``etm-preprocess``.  This
module combines three responsibilities that are tightly coupled at this stage:

* scan sequence JSONL files and build per-field vocabularies;
* encode token fields into compact memmap arrays;
* expose a Keras Sequence that samples history windows within each segment.

Each encoded segment is preceded by one all-PAD boundary row.  The training
target skips token[0] of each segment by selecting only non-PAD rows whose
previous row is also non-PAD.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    from tensorflow.keras.utils import Sequence as KerasSequence
except Exception:  # pragma: no cover - TensorFlow is only required for training.
    class KerasSequence:  # type: ignore[no-redef]
        pass


PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"
UNKNOWN_TOKEN = "<unknown>"
PAD_ID = 0
OOV_ID = 1

STALE_INDEX_FILES = ("examples.dat", "segment_starts.dat", "segment_lengths.dat")

INPUT_FIELDS = (
    "entry_func",
    "entry_off",
    "src_ctrl_func",
    "src_ctrl_off",
    "ctrl_type",
    "dst_node",
    "icount",
)

TARGET_FIELDS = {
    "next_dst_node": "dst_node",
    "next_ctrl_type": "ctrl_type",
}

VOCAB_SOURCE_FIELDS = {
    "function": ("entry_func", "src_ctrl_func"),
    "offset": ("entry_off", "src_ctrl_off"),
    "node": ("dst_node",),
    "ctrl_type": ("ctrl_type",),
}


def clean_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text == UNKNOWN_TOKEN:
        return None
    return text


def parse_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text or text == UNKNOWN_TOKEN:
            return None
        try:
            return int(text, 0)
        except ValueError:
            return None
    return None


def instruction_count_id(value: Any, clip: int) -> int:
    parsed = parse_int(value)
    if parsed is None or parsed < 0:
        return OOV_ID
    return min(parsed, clip) + 2


def make_node(function: Optional[str], offset: Optional[str]) -> Optional[str]:
    if function is None or offset is None:
        return None
    return f"{function}@{offset}"


def ctrl_type(kind: Any, atom: Any) -> Optional[str]:
    kind_text = clean_value(kind)
    if kind_text is None:
        return None

    atom_text = clean_value(atom)
    if atom_text == "E":
        outcome = "taken"
    elif atom_text == "N":
        outcome = "not_taken"
    elif atom_text is None:
        outcome = "unknown"
    else:
        outcome = atom_text.lower()
    return f"{kind_text}:{outcome}"


@dataclass
class DatasetBuildConfig:
    max_function_vocab: int = 50000
    max_offset_vocab: int = 65536
    max_node_vocab: int = 200000
    max_ctrl_type_vocab: int = 0
    instruction_count_clip: int = 255
    min_segment_length: int = 2
    recursive: bool = False
    patterns: Tuple[str, ...] = ("*.jsonl",)

    def validate(self) -> None:
        vocab_limits = {
            "max_function_vocab": self.max_function_vocab,
            "max_offset_vocab": self.max_offset_vocab,
            "max_node_vocab": self.max_node_vocab,
            "max_ctrl_type_vocab": self.max_ctrl_type_vocab,
        }
        for name, limit in vocab_limits.items():
            if limit < 0 or limit == 1:
                raise ValueError(f"{name} must be 0 (unlimited) or at least 2")
        if self.instruction_count_clip < 0:
            raise ValueError("instruction_count_clip must be >= 0")
        if self.min_segment_length < 2:
            raise ValueError("min_segment_length must be >= 2")
        if not self.patterns or any(not pattern.strip() for pattern in self.patterns):
            raise ValueError("patterns must contain at least one non-empty glob")


class FieldVocab:
    def __init__(self, name: str, max_size: int = 0) -> None:
        if max_size < 0 or max_size == 1:
            raise ValueError("max_size must be 0 (unlimited) or at least 2")
        self.name = name
        self.max_size = max_size
        self.token_to_id: Dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            OOV_TOKEN: OOV_ID,
        }
        self.id_to_token: List[str] = [PAD_TOKEN, OOV_TOKEN]

    def build(self, counter: Counter[str]) -> None:
        if self.max_size > 0:
            limit = self.max_size - len(self.id_to_token)
            items = counter.most_common(limit)
        else:
            items = counter.most_common()

        for token, _ in items:
            if token in self.token_to_id:
                continue
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

    def encode(self, value: Any) -> int:
        token = clean_value(value)
        if token is None:
            return OOV_ID
        return self.token_to_id.get(token, OOV_ID)

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_size": self.max_size,
            "tokens": self.id_to_token,
        }

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "FieldVocab":
        if not isinstance(record, dict):
            raise ValueError("vocab record must be an object")
        vocab = cls(str(record["name"]), int(record.get("max_size") or 0))
        tokens = record.get("tokens")
        if not isinstance(tokens, list) or len(tokens) < 2:
            raise ValueError(f"invalid vocab record for {vocab.name}")
        vocab.id_to_token = [str(token) for token in tokens]
        if vocab.id_to_token[:2] != [PAD_TOKEN, OOV_TOKEN]:
            raise ValueError(f"invalid reserved tokens for {vocab.name}")
        if len(set(vocab.id_to_token)) != len(vocab.id_to_token):
            raise ValueError(f"duplicate tokens in vocab {vocab.name}")
        vocab.token_to_id = {token: idx for idx, token in enumerate(vocab.id_to_token)}
        return vocab


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc.msg}") from exc
            if isinstance(value, dict):
                yield value


def discover_sequence_files(
    paths: Sequence[Path],
    *,
    patterns: Sequence[str],
    recursive: bool,
) -> List[Path]:
    result = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if path.is_file():
            result.add(path.resolve())
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"input path does not exist: {path}")

        root = path.resolve()
        for pattern in patterns:
            iterator = root.rglob(pattern) if recursive else root.glob(pattern)
            for item in iterator:
                if item.is_file():
                    result.add(item.resolve())
    return sorted(result, key=str)


def location_value(location: Dict[str, Any], key: str) -> Optional[str]:
    if not isinstance(location, dict):
        return None
    return clean_value(location.get(key))


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def iter_segment_edges(record: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    tokens = record.get("tokens")
    if not isinstance(tokens, list):
        return

    previous_dst = as_dict(record.get("start"))
    for token in tokens:
        if not isinstance(token, dict):
            continue
        src_ctrl = as_dict(token.get("src_ctrl"))
        ctrl = as_dict(token.get("ctrl"))
        dst = as_dict(token.get("dst"))
        dst_func = location_value(dst, "function")
        dst_off = location_value(dst, "function_offset")

        yield {
            "entry_func": location_value(previous_dst, "function"),
            "entry_off": location_value(previous_dst, "function_offset"),
            "src_ctrl_func": location_value(src_ctrl, "function"),
            "src_ctrl_off": location_value(src_ctrl, "function_offset"),
            "ctrl_type": ctrl_type(ctrl.get("kind"), ctrl.get("atom")),
            "dst_node": make_node(dst_func, dst_off),
            "icount_raw": token.get("instruction_count"),
        }
        previous_dst = dst


def scan_sequences(
    input_files: Sequence[Path],
    config: DatasetBuildConfig,
) -> Tuple[Dict[str, Counter[str]], Dict[str, Any]]:
    counters = {name: Counter() for name in VOCAB_SOURCE_FIELDS}
    stats: Dict[str, Any] = {
        "input_files": len(input_files),
        "segments_seen": 0,
        "segments_kept": 0,
        "segments_dropped_short": 0,
        "edges": 0,
        "examples": 0,
    }

    with tqdm(desc="scan segments", unit="segment") as progress:
        for path in input_files:
            progress.set_postfix(file=path.name[:40])
            for record in iter_jsonl(path):
                edges = list(iter_segment_edges(record))
                stats["segments_seen"] += 1
                progress.update(1)
                if stats["segments_seen"] % 1000 == 0:
                    progress.set_postfix(
                        file=path.name[:30],
                        kept=stats["segments_kept"],
                        edges=stats["edges"],
                    )
                if len(edges) < config.min_segment_length:
                    stats["segments_dropped_short"] += 1
                    continue

                stats["segments_kept"] += 1
                stats["edges"] += len(edges)
                stats["examples"] += len(edges) - 1
                for edge in edges:
                    for vocab_name, source_fields in VOCAB_SOURCE_FIELDS.items():
                        counters[vocab_name].update(
                            edge[field] for field in source_fields if edge[field] is not None
                        )

    return counters, stats


def build_vocabs(counters: Dict[str, Counter[str]], config: DatasetBuildConfig) -> Dict[str, FieldVocab]:
    vocabs = {
        "function": FieldVocab("function", config.max_function_vocab),
        "offset": FieldVocab("offset", config.max_offset_vocab),
        "node": FieldVocab("node", config.max_node_vocab),
        "ctrl_type": FieldVocab("ctrl_type", config.max_ctrl_type_vocab),
    }
    for name, vocab in vocabs.items():
        vocab.build(counters[name])
    return vocabs


def save_vocabs(path: Path, vocabs: Dict[str, FieldVocab]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            {name: vocab.to_dict() for name, vocab in vocabs.items()},
            stream,
            ensure_ascii=False,
            indent=2,
        )


def load_vocabs(dataset_dir: Path) -> Dict[str, FieldVocab]:
    with (dataset_dir / "vocab.json").open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("vocab.json must contain an object")

    vocabs = {}
    for name, record in raw.items():
        vocab = FieldVocab.from_dict(record)
        if vocab.name != name:
            raise ValueError(f"vocab name mismatch: key {name!r}, record {vocab.name!r}")
        vocabs[name] = vocab
    return vocabs


def encode_edge(
    edge: Dict[str, Any],
    vocabs: Dict[str, FieldVocab],
    config: DatasetBuildConfig,
) -> Dict[str, int]:
    return {
        "entry_func": vocabs["function"].encode(edge.get("entry_func")),
        "entry_off": vocabs["offset"].encode(edge.get("entry_off")),
        "src_ctrl_func": vocabs["function"].encode(edge.get("src_ctrl_func")),
        "src_ctrl_off": vocabs["offset"].encode(edge.get("src_ctrl_off")),
        "ctrl_type": vocabs["ctrl_type"].encode(edge.get("ctrl_type")),
        "dst_node": vocabs["node"].encode(edge.get("dst_node")),
        "icount": instruction_count_id(edge.get("icount_raw"), config.instruction_count_clip),
    }


def remove_stale_index_files(output_dir: Path) -> None:
    for name in STALE_INDEX_FILES:
        (output_dir / name).unlink(missing_ok=True)


def encode_sequences(
    input_files: Sequence[Path],
    output_dir: Path,
    vocabs: Dict[str, FieldVocab],
    config: DatasetBuildConfig,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    total_edges = int(stats["edges"])
    total_examples = int(stats["examples"])
    total_segments = int(stats["segments_kept"])
    total_rows = total_edges + total_segments

    field_arrays = {
        field: np.memmap(
            output_dir / f"{field}.dat",
            dtype="uint32",
            mode="w+",
            shape=(total_rows,),
        )
        for field in INPUT_FIELDS
    }
    for array in field_arrays.values():
        array[:] = PAD_ID

    row_ptr = 0
    edge_ptr = 0
    segment_ptr = 0

    with tqdm(total=int(stats["segments_seen"]), desc="encode segments", unit="segment") as progress:
        for path in input_files:
            progress.set_postfix(file=path.name[:40])
            for record in iter_jsonl(path):
                progress.update(1)
                edges = list(iter_segment_edges(record))
                if len(edges) < config.min_segment_length:
                    continue

                row_ptr += 1

                for local_pos, edge in enumerate(edges):
                    encoded = encode_edge(edge, vocabs, config)
                    for field in INPUT_FIELDS:
                        field_arrays[field][row_ptr + local_pos] = encoded[field]

                row_ptr += len(edges)
                edge_ptr += len(edges)
                segment_ptr += 1
                if segment_ptr % 1000 == 0:
                    progress.set_postfix(
                        file=path.name[:30],
                        kept=segment_ptr,
                        edges=edge_ptr,
                    )

    for array in field_arrays.values():
        array.flush()

    if row_ptr != total_rows or edge_ptr != total_edges or segment_ptr != total_segments:
        raise RuntimeError(
            "encoded counts do not match scan counts: "
            f"rows {row_ptr}/{total_rows}, "
            f"edges {edge_ptr}/{total_edges}, "
            f"segments {segment_ptr}/{total_segments}"
        )

    meta = {
        "input_fields": list(INPUT_FIELDS),
        "target_fields": TARGET_FIELDS,
        "total_rows": total_rows,
        "total_edges": total_edges,
        "total_segments": total_segments,
        "total_examples": total_examples,
        "field_dtype": "uint32",
        "boundary_row": "all input fields are PAD_ID",
        "instruction_count_clip": config.instruction_count_clip,
        "instruction_count_vocab_size": config.instruction_count_clip + 3,
        "min_segment_length": config.min_segment_length,
        "vocab_sizes": {
            "function": vocabs["function"].size,
            "offset": vocabs["offset"].size,
            "node": vocabs["node"].size,
            "ctrl_type": vocabs["ctrl_type"].size,
            "icount": config.instruction_count_clip + 3,
        },
        "scan_stats": stats,
        "source_files": [str(path) for path in input_files],
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)
    return meta


def build_dataset(
    input_paths: Sequence[Path],
    output_dir: Path,
    config: DatasetBuildConfig,
) -> Dict[str, Any]:
    config.validate()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"dataset output path exists but is not a directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_index_files(output_dir)

    input_files = discover_sequence_files(
        input_paths,
        patterns=config.patterns,
        recursive=config.recursive,
    )
    if not input_files:
        raise FileNotFoundError("no input sequence JSONL files found")

    counters, stats = scan_sequences(input_files, config)
    if stats["examples"] <= 0:
        raise ValueError("no training examples found; need segments with at least two tokens")

    vocabs = build_vocabs(counters, config)
    (output_dir / "meta.json").unlink(missing_ok=True)
    save_vocabs(output_dir / "vocab.json", vocabs)
    return encode_sequences(input_files, output_dir, vocabs, config, stats)


def load_meta(dataset_dir: Path) -> Dict[str, Any]:
    with (dataset_dir / "meta.json").open("r", encoding="utf-8") as stream:
        meta = json.load(stream)
    if not isinstance(meta, dict):
        raise ValueError("meta.json must contain an object")
    return meta


def load_vocab_sizes(dataset_dir: Path) -> Dict[str, int]:
    meta = load_meta(dataset_dir)
    sizes = meta.get("vocab_sizes")
    if not isinstance(sizes, dict):
        raise ValueError("meta.json missing vocab_sizes")
    return {key: int(value) for key, value in sizes.items()}


class FlowSequence(KerasSequence):
    """Keras generator that samples next-token targets without crossing segments."""

    def __init__(
        self,
        dataset_dir: Path,
        *,
        max_seq_len: int,
        batch_size: int,
        mode: str = "train",
        validation_split: float = 0.1,
        shuffle: bool = True,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0.0, 1.0)")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.dataset_dir = dataset_dir.expanduser().resolve()
        self.meta = load_meta(self.dataset_dir)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle and mode == "train"
        self.rng = np.random.default_rng(seed)

        total_rows = int(self.meta["total_rows"])
        field_dtype = str(self.meta.get("field_dtype", "uint32"))

        self.fields = {
            field: np.memmap(
                self.dataset_dir / f"{field}.dat",
                dtype=field_dtype,
                mode="r",
                shape=(total_rows,),
            )
            for field in INPUT_FIELDS
        }

        ctrl_type_values = np.asarray(self.fields["ctrl_type"])
        valid_targets = np.nonzero(
            (ctrl_type_values[1:] != PAD_ID) & (ctrl_type_values[:-1] != PAD_ID)
        )[0] + 1
        split_row = self.choose_split_row(ctrl_type_values, validation_split)
        if mode == "train":
            selected = valid_targets[valid_targets < split_row]
        else:
            selected = valid_targets[valid_targets >= split_row]
        self.indices = selected.astype(np.int64, copy=False)
        self.on_epoch_end()

    @staticmethod
    def choose_split_row(ctrl_type_values: np.ndarray, validation_split: float) -> int:
        if validation_split <= 0.0:
            return len(ctrl_type_values)

        raw_split = int(len(ctrl_type_values) * (1.0 - validation_split))
        boundaries = np.nonzero(ctrl_type_values == PAD_ID)[0]
        candidates = np.append(boundaries, len(ctrl_type_values))
        distances = np.abs(candidates - raw_split)
        return int(candidates[np.where(distances == distances.min())[0][-1]])

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self) -> None:
        if self.shuffle and len(self.indices) > 1:
            self.rng.shuffle(self.indices)

    def __getitem__(self, batch_index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(f"batch index out of range: {batch_index}")
        batch_ids = self.indices[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        batch_size = len(batch_ids)
        x = {
            field: np.zeros((batch_size, self.max_seq_len), dtype=np.int32)
            for field in INPUT_FIELDS
        }
        y = {
            target: np.empty((batch_size,), dtype=np.int32)
            for target in TARGET_FIELDS
        }

        for row, example_id in enumerate(batch_ids):
            target_abs = int(example_id)

            source_start = max(0, target_abs - self.max_seq_len)
            boundary_offsets = np.flatnonzero(
                self.fields["ctrl_type"][source_start:target_abs] == PAD_ID
            )
            if len(boundary_offsets):
                source_start += int(boundary_offsets[-1]) + 1

            history_len = target_abs - source_start
            source_slice = slice(source_start, target_abs)
            target_slice = slice(self.max_seq_len - history_len, self.max_seq_len)
            for field in INPUT_FIELDS:
                x[field][row, target_slice] = self.fields[field][source_slice]

            for output_name, source_field in TARGET_FIELDS.items():
                y[output_name][row] = self.fields[source_field][target_abs]

        return x, y


def dataset_ready(dataset_dir: Path) -> bool:
    dataset_dir = dataset_dir.expanduser()
    required = ["meta.json", "vocab.json", *(f"{field}.dat" for field in INPUT_FIELDS)]
    if not all((dataset_dir / item).is_file() for item in required):
        return False
    try:
        meta = load_meta(dataset_dir)
        total_rows = int(meta["total_rows"])
        field_dtype = np.dtype(str(meta.get("field_dtype", "uint32")))
        if total_rows <= 0 or tuple(meta.get("input_fields", ())) != INPUT_FIELDS:
            return False
        if field_dtype != np.dtype("uint32"):
            return False

        expected_bytes = total_rows * field_dtype.itemsize
        if any(
            (dataset_dir / f"{field}.dat").stat().st_size != expected_bytes
            for field in INPUT_FIELDS
        ):
            return False

        vocabs = load_vocabs(dataset_dir)
        vocab_sizes = meta.get("vocab_sizes")
        if not isinstance(vocab_sizes, dict):
            return False
        for name in VOCAB_SOURCE_FIELDS:
            if name not in vocabs or int(vocab_sizes.get(name, -1)) != vocabs[name].size:
                return False
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a memmap flow dataset from sequence JSONL files.")
    parser.add_argument("--input", "-i", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        help="Output dataset directory. Existing directories are allowed.",
    )
    parser.add_argument("--glob", dest="patterns", action="append", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-function-vocab", type=int, default=50000)
    parser.add_argument("--max-offset-vocab", type=int, default=65536)
    parser.add_argument("--max-node-vocab", type=int, default=200000)
    parser.add_argument("--max-ctrl-type-vocab", type=int, default=0)
    parser.add_argument("--instruction-count-clip", type=int, default=255)
    parser.add_argument("--min-segment-length", type=int, default=2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = DatasetBuildConfig(
        max_function_vocab=args.max_function_vocab,
        max_offset_vocab=args.max_offset_vocab,
        max_node_vocab=args.max_node_vocab,
        max_ctrl_type_vocab=args.max_ctrl_type_vocab,
        instruction_count_clip=args.instruction_count_clip,
        min_segment_length=args.min_segment_length,
        recursive=args.recursive,
        patterns=tuple(args.patterns or ["*.jsonl"]),
    )
    try:
        meta = build_dataset(args.input, args.output_dir, config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"dataset: {args.output_dir}")
    print(f"segments: {meta['total_segments']}")
    print(f"edges: {meta['total_edges']}")
    print(f"examples: {meta['total_examples']}")
    print(f"vocab_sizes: {meta['vocab_sizes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
