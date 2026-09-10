#!/usr/bin/env python3
"""Score flow-sequence datasets with a trained next-edge model.

This script matches the current dataset format produced by ``etm-build-dataset``:
one uint32 memmap per input field, PAD rows between segments, and targets at
non-PAD rows whose previous row is also non-PAD.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.models import load_model
from tqdm import tqdm

from etm_flow.data.dataset import INPUT_FIELDS, PAD_ID
from etm_flow.modeling.models.common import LearnedPositionEmbedding


def load_config(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    predict = raw["predict"]
    return SimpleNamespace(
        dataset_dir=Path(predict["dataset_dir"]),
        model_path=Path(predict["model"]),
        max_seq_len=int(predict["max_seq_len"]),
        batch_size=int(predict["batch_size"]),
        output=Path(predict["output"]),
        prob_threshold=float(predict["prob_threshold"]),
        window_size=int(predict["window_size"]),
        window_low_threshold=int(predict["window_low_threshold"]),
        score_mode=str(predict["score_mode"]),
        ctrl_weight=float(predict["ctrl_weight"]),
        partial_window=bool(predict["partial_window"]),
        write_all_scores=bool(predict["write_all_scores"]),
        limit=int(predict["limit"]),
        show_progress=bool(predict["show_progress"]),
    )


def configure_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_flow_model(model_path: Path) -> Any:
    custom_objects = {
        "LearnedPositionEmbedding": LearnedPositionEmbedding,
        "FlowModels>LearnedPositionEmbedding": LearnedPositionEmbedding,
    }
    return load_model(str(model_path), custom_objects=custom_objects, compile=False)


def load_vocab_tokens(dataset_dir: Path) -> Dict[str, List[str]]:
    with (dataset_dir / "vocab.json").open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("vocab.json must contain an object")

    result: Dict[str, List[str]] = {}
    for name, record in raw.items():
        if not isinstance(record, dict) or not isinstance(record.get("tokens"), list):
            raise ValueError(f"invalid vocab record: {name}")
        result[name] = [str(token) for token in record["tokens"]]
    return result


def token_name(vocabs: Dict[str, List[str]], vocab_name: str, token_id: int) -> str:
    tokens = vocabs.get(vocab_name, [])
    if 0 <= token_id < len(tokens):
        return tokens[token_id]
    return f"<id:{token_id}>"


class EncodedFlowDataset:
    def __init__(self, dataset_dir: Path) -> None:
        self.np = np
        self.input_fields = tuple(INPUT_FIELDS)
        self.pad_id = int(PAD_ID)
        self.dataset_dir = dataset_dir.expanduser().resolve()

        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as stream:
            self.meta = json.load(stream)
        self.total_rows = int(self.meta["total_rows"])
        self.field_dtype = str(self.meta.get("field_dtype", "uint32"))

        self.fields = {
            field: np.memmap(
                self.dataset_dir / f"{field}.dat",
                dtype=self.field_dtype,
                mode="r",
                shape=(self.total_rows,),
            )
            for field in self.input_fields
        }
        ctrl_type_values = np.asarray(self.fields["ctrl_type"])
        self.valid_targets = np.nonzero(
            (ctrl_type_values[1:] != self.pad_id) & (ctrl_type_values[:-1] != self.pad_id)
        )[0] + 1
        self.vocabs = load_vocab_tokens(self.dataset_dir)

    def make_batch(self, target_rows: Sequence[int], max_seq_len: int) -> Dict[str, Any]:
        np = self.np
        batch_size = len(target_rows)
        x = {
            field: np.zeros((batch_size, max_seq_len), dtype=np.int32)
            for field in self.input_fields
        }

        for row, target_abs_raw in enumerate(target_rows):
            target_abs = int(target_abs_raw)
            history_len = min(max_seq_len, target_abs)
            if history_len <= 0:
                continue

            source_start = target_abs - history_len
            source_slice = slice(source_start, target_abs)
            target_slice = slice(max_seq_len - history_len, max_seq_len)
            for field in self.input_fields:
                x[field][row, target_slice] = self.fields[field][source_slice]

            boundary_positions = np.flatnonzero(x["ctrl_type"][row] == self.pad_id)
            if len(boundary_positions):
                last_boundary = int(boundary_positions[-1])
                for field in self.input_fields:
                    x[field][row, : last_boundary + 1] = self.pad_id

        return x

    def target_ids(self, target_rows: Sequence[int]) -> Tuple[Any, Any]:
        rows = self.np.asarray(target_rows, dtype=self.np.int64)
        return (
            self.np.asarray(self.fields["dst_node"][rows], dtype=self.np.int64),
            self.np.asarray(self.fields["ctrl_type"][rows], dtype=self.np.int64),
        )


def validate_model(model: Any, dataset: "EncodedFlowDataset", max_seq_len: int) -> None:
    """校验模型的输入字段/序列长度/输出维度与数据集一致。

    predict.max_seq_len 与训练时固定的输入长度、以及输出层维度和词表必须匹配，
    否则要么在 predict 时抛出难以理解的形状错误，要么静默给出错误的打分。
    """
    input_names = {tensor.name.split(":", 1)[0].split("/")[-1] for tensor in model.inputs}
    if input_names != set(INPUT_FIELDS):
        raise ValueError(
            f"model inputs {sorted(input_names)} do not match dataset fields {sorted(INPUT_FIELDS)}"
        )

    sequence_lengths = {tensor.shape[1] for tensor in model.inputs}
    if len(sequence_lengths) != 1 or None in sequence_lengths:
        raise ValueError(f"model inputs must have one fixed sequence length: {sequence_lengths}")
    model_seq_len = int(next(iter(sequence_lengths)))
    if model_seq_len != max_seq_len:
        raise ValueError(
            f"predict.max_seq_len ({max_seq_len}) does not match the model's fixed "
            f"input length ({model_seq_len}); set predict.max_seq_len to {model_seq_len}"
        )

    vocab_sizes = {
        name: len(tokens) for name, tokens in dataset.vocabs.items()
    }
    output_sizes = {
        name: int(tensor.shape[-1])
        for name, tensor in zip(model.output_names, model.outputs)
    }
    expected = {
        "next_dst_node": int(vocab_sizes.get("node", -1)),
        "next_ctrl_type": int(vocab_sizes.get("ctrl_type", -1)),
    }
    if output_sizes != expected:
        raise ValueError(
            f"model outputs {output_sizes} do not match dataset vocabularies {expected}"
        )


def output_dict(model: Any, predictions: Any) -> Dict[str, Any]:
    if isinstance(predictions, dict):
        return predictions
    if isinstance(predictions, (list, tuple)):
        return {name: value for name, value in zip(model.output_names, predictions)}
    raise TypeError(f"unsupported model prediction type: {type(predictions).__name__}")


def probability_at(prob_matrix: Any, row: int, token_id: int) -> float:
    if token_id < 0 or token_id >= int(prob_matrix.shape[1]):
        return 0.0
    return float(prob_matrix[row, token_id])


def iter_scores(
    model: Any,
    dataset: EncodedFlowDataset,
    *,
    max_seq_len: int,
    batch_size: int,
    limit: int,
    score_mode: str,
    ctrl_weight: float,
    show_progress: bool,
) -> Iterator[Dict[str, Any]]:
    target_rows = dataset.valid_targets
    if limit > 0:
        target_rows = target_rows[:limit]

    batch_starts = range(0, len(target_rows), batch_size)
    if show_progress:
        batch_starts = tqdm(batch_starts, desc="predict", unit="batch")  # type: ignore[assignment]

    sample_base = 0
    for batch_start in batch_starts:
        batch_rows = target_rows[batch_start : batch_start + batch_size]
        x = dataset.make_batch(batch_rows, max_seq_len)
        true_dst_ids, true_ctrl_ids = dataset.target_ids(batch_rows)

        raw_predictions = model(x, training=False)
        predictions = output_dict(model, raw_predictions)
        # model(x) 返回 eager TF 张量；一次性转成 numpy，后续逐行索引才不会
        # 每次触发单元素 gather + 主机同步。
        dst_probs = np.asarray(predictions["next_dst_node"])
        ctrl_probs = np.asarray(predictions["next_ctrl_type"])

        pred_dst_ids = np.argmax(dst_probs, axis=1)
        pred_ctrl_ids = np.argmax(ctrl_probs, axis=1)

        for row, target_abs_raw in enumerate(batch_rows):
            true_dst_id = int(true_dst_ids[row])
            true_ctrl_id = int(true_ctrl_ids[row])
            dst_prob = probability_at(dst_probs, row, true_dst_id)
            ctrl_prob = probability_at(ctrl_probs, row, true_ctrl_id)
            if score_mode == "combined":
                score_prob = float(dst_prob * (ctrl_prob ** ctrl_weight))
            else:
                score_prob = dst_prob

            yield {
                "sample_index": sample_base + row,
                "target_row": int(target_abs_raw),
                "dst_node_id": true_dst_id,
                "dst_node": token_name(dataset.vocabs, "node", true_dst_id),
                "dst_prob": dst_prob,
                "pred_dst_node_id": int(pred_dst_ids[row]),
                "pred_dst_node": token_name(dataset.vocabs, "node", int(pred_dst_ids[row])),
                "ctrl_type_id": true_ctrl_id,
                "ctrl_type": token_name(dataset.vocabs, "ctrl_type", true_ctrl_id),
                "ctrl_prob": ctrl_prob,
                "pred_ctrl_type_id": int(pred_ctrl_ids[row]),
                "pred_ctrl_type": token_name(dataset.vocabs, "ctrl_type", int(pred_ctrl_ids[row])),
                "score_mode": score_mode,
                "score_prob": score_prob,
            }
        sample_base += len(batch_rows)


def should_alarm(
    window: Deque[Tuple[int, bool, float]],
    *,
    low_count: int,
    low_threshold: int,
    full_window_size: int,
    partial: bool,
) -> bool:
    if not partial and len(window) < full_window_size:
        return False
    return low_count >= low_threshold


def write_report(cfg: SimpleNamespace) -> Dict[str, Any]:
    dataset_dir = cfg.dataset_dir.expanduser().resolve()
    model_path = cfg.model_path.expanduser().resolve()
    max_seq_len = int(cfg.max_seq_len)
    batch_size = int(cfg.batch_size)
    ctrl_weight = float(cfg.ctrl_weight)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    configure_gpu_memory_growth()
    dataset = EncodedFlowDataset(dataset_dir)
    model = load_flow_model(model_path)
    validate_model(model, dataset, max_seq_len)

    output_path = cfg.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    low_samples = 0
    alarm_windows = 0
    max_window_low_count = 0
    previous_target_row: Optional[int] = None
    window: Deque[Tuple[int, bool, float]] = deque(maxlen=int(cfg.window_size))

    with output_path.open("w", encoding="utf-8") as stream:
        for record in iter_scores(
            model,
            dataset,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            limit=int(cfg.limit),
            score_mode=str(cfg.score_mode),
            ctrl_weight=ctrl_weight,
            show_progress=bool(cfg.show_progress),
        ):
            target_row = int(record["target_row"])
            if previous_target_row is None or target_row != previous_target_row + 1:
                window.clear()
            previous_target_row = target_row

            total_samples += 1
            is_low = float(record["score_prob"]) < float(cfg.prob_threshold)
            if is_low:
                low_samples += 1

            window.append((target_row, is_low, float(record["score_prob"])))
            window_low_count = sum(1 for _, item_is_low, _ in window if item_is_low)
            max_window_low_count = max(max_window_low_count, window_low_count)
            record.update(
                {
                    "event": "score" if cfg.write_all_scores else "low_token",
                    "is_low_prob": is_low,
                    "prob_threshold": float(cfg.prob_threshold),
                    "window_size": int(cfg.window_size),
                    "window_low_count": window_low_count,
                }
            )

            if cfg.write_all_scores or is_low:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            if should_alarm(
                window,
                low_count=window_low_count,
                low_threshold=int(cfg.window_low_threshold),
                full_window_size=int(cfg.window_size),
                partial=bool(cfg.partial_window),
            ):
                alarm_windows += 1
                alarm_record = {
                    "event": "window_alarm",
                    "window_start_row": int(window[0][0]),
                    "window_end_row": int(window[-1][0]),
                    "window_observed_size": len(window),
                    "window_size": int(cfg.window_size),
                    "window_low_threshold": int(cfg.window_low_threshold),
                    "window_low_count": window_low_count,
                    "prob_threshold": float(cfg.prob_threshold),
                    "score_mode": str(cfg.score_mode),
                    "end_sample_index": int(record["sample_index"]),
                    "end_dst_node": record["dst_node"],
                    "end_score_prob": float(record["score_prob"]),
                }
                stream.write(json.dumps(alarm_record, ensure_ascii=False) + "\n")

    low_ratio = (low_samples / total_samples) if total_samples else 0.0
    return {
        "dataset_dir": str(dataset_dir),
        "model": str(model_path),
        "output": str(output_path),
        "total_samples": total_samples,
        "low_samples": low_samples,
        "low_ratio": low_ratio,
        "alarm_windows": alarm_windows,
        "max_window_low_count": max_window_low_count,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "prob_threshold": float(cfg.prob_threshold),
        "window_size": int(cfg.window_size),
        "window_low_threshold": int(cfg.window_low_threshold),
        "score_mode": str(cfg.score_mode),
        "ctrl_weight": ctrl_weight,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict and score encoded flow sequences.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config YAML file. Default: configs/default.yaml",
    )
    return parser


def validate_config(cfg: SimpleNamespace) -> None:
    if int(cfg.max_seq_len) <= 0:
        raise ValueError("predict.max_seq_len must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("predict.batch_size must be > 0")
    if not 0.0 <= float(cfg.prob_threshold) <= 1.0:
        raise ValueError("predict.prob_threshold must be in [0, 1]")
    if int(cfg.window_size) <= 0:
        raise ValueError("predict.window_size must be > 0")
    if int(cfg.window_low_threshold) <= 0:
        raise ValueError("predict.window_low_threshold must be > 0")
    if int(cfg.limit) < 0:
        raise ValueError("predict.limit must be >= 0")
    if float(cfg.ctrl_weight) < 0:
        raise ValueError("predict.ctrl_weight must be >= 0")
    if str(cfg.score_mode) not in {"dst", "combined"}:
        raise ValueError("predict.score_mode must be 'dst' or 'combined'")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        validate_config(cfg)
        summary = write_report(cfg)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    print("prediction complete")
    print(f"dataset: {summary['dataset_dir']}")
    print(f"model: {summary['model']}")
    print(f"samples: {summary['total_samples']}")
    print(f"low_samples: {summary['low_samples']} ({summary['low_ratio']:.4%})")
    print(f"alarm_windows: {summary['alarm_windows']}")
    print(f"max_window_low_count: {summary['max_window_low_count']}")
    print(f"report: {summary['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
