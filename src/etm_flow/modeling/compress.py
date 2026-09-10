#!/usr/bin/env python3
"""Prune and fine-tune a trained flow model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

from etm_flow.data.dataset import INPUT_FIELDS, FlowSequence, load_vocab_sizes
from etm_flow.modeling.models.common import LearnedPositionEmbedding
from etm_flow.modeling.train import compile_model



def load_config(path: Path) -> SimpleNamespace:

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    compress = raw["compress"]
    dataset = raw["dataset"]
    training = raw["training"]
    loss_weights = raw["loss_weights"]
    return SimpleNamespace(
        config_file=path.expanduser().resolve(),
        dataset_dir=Path(dataset["dir"]),
        input_model=Path(compress["input_model"]),
        output_model=Path(compress["output_model"]),
        batch_size=int(compress["batch_size"]),
        validation_split=float(training["validation_split"]),
        seed=int(training["seed"]),
        epochs=int(compress["epochs"]),
        recovery_epochs=int(compress["recovery_epochs"]),
        learning_rate=float(compress["learning_rate"]),
        initial_sparsity=float(compress["initial_sparsity"]),
        final_sparsity=float(compress["final_sparsity"]),
        pruning_updates=int(compress["pruning_updates"]),
        exclude_patterns=tuple(str(item).lower() for item in compress["exclude_patterns"]),
        loss_weight_dst_node=float(loss_weights["dst_node"]),
        loss_weight_ctrl_type=float(loss_weights["ctrl_type"]),
    )


def validate_config(cfg: SimpleNamespace) -> None:
    if not cfg.dataset_dir.is_dir():
        raise ValueError(f"dataset directory not found: {cfg.dataset_dir}")
    if not cfg.input_model.is_file():
        raise ValueError(f"input model not found: {cfg.input_model}")
    if cfg.batch_size <= 0 or cfg.epochs <= 0:
        raise ValueError("batch_size and epochs must be > 0")
    if not 0 <= cfg.recovery_epochs < cfg.epochs:
        raise ValueError("recovery_epochs must be in [0, epochs)")
    if not 0.0 <= cfg.validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1)")
    if not 0.0 <= cfg.initial_sparsity < cfg.final_sparsity < 1.0:
        raise ValueError("sparsity must satisfy 0 <= initial_sparsity < final_sparsity < 1")
    if cfg.pruning_updates <= 0:
        raise ValueError("pruning_updates must be > 0")

    paths = {cfg.input_model.resolve(), cfg.output_model.resolve()}
    if len(paths) != 2:
        raise ValueError("input_model and output_model must be different paths")
    if cfg.output_model.suffix != ".keras":
        raise ValueError("output_model must end with .keras")


def configure_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_flow_model(path: Path) -> Any:
    custom_objects = {
        "LearnedPositionEmbedding": LearnedPositionEmbedding,
        "FlowModels>LearnedPositionEmbedding": LearnedPositionEmbedding,
    }
    return load_model(str(path), custom_objects=custom_objects, compile=False)


def selected_weights(model: Any, exclude_patterns: Sequence[str]) -> List[Any]:
    result = []
    for weight in model.trainable_weights:
        name = str(getattr(weight, "path", getattr(weight, "name", ""))).lower()
        if (
            len(weight.shape) >= 2
            and tf.as_dtype(weight.dtype).is_floating
            and not any(pattern in name for pattern in exclude_patterns)
        ):
            result.append(weight)
    return result


def sparsity_stats(weights: Iterable[Any]) -> Dict[str, Any]:
    total = 0
    zeros = 0
    tensors = 0
    for weight in weights:
        values = np.asarray(weight.numpy())
        total += int(values.size)
        zeros += int(np.count_nonzero(values == 0))
        tensors += 1
    return {
        "tensors": tensors,
        "parameters": total,
        "zero_parameters": zeros,
        "sparsity": (zeros / total) if total else 0.0,
    }


def build_pruning_callback(
    *,
    initial_sparsity: float,
    final_sparsity: float,
    end_step: int,
    frequency: int,
    exclude_patterns: Sequence[str],
) -> Any:
    """Build a gradual global magnitude-pruning callback."""
    class MagnitudePruningCallback(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.weight_masks: List[Tuple[Any, Any]] = []
            self.last_sparsity = 0.0
            self.actual_sparsity = 0.0
            self.parameter_count = 0
            self.zero_count = 0
            self._apply_compiled: Any = None

        def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
            weights = selected_weights(self.model, exclude_patterns)
            if not weights:
                raise ValueError("the model contains no eligible weights for pruning")
            self.weight_masks = [
                (
                    weight,
                    tf.Variable(tf.ones(weight.shape, dtype=tf.bool), trainable=False),
                )
                for weight in weights
            ]

            @tf.function
            def apply_masks() -> None:
                for weight, mask in self.weight_masks:
                    weight.assign(tf.where(mask, weight, tf.cast(0, weight.dtype)))

            self._apply_compiled = apply_masks
            self.update_masks(initial_sparsity)
            self.apply_masks()
            parameters = sum(int(np.prod(weight.shape)) for weight in weights)
            print(f"prunable tensors: {len(weights)}")
            print(f"prunable parameters: {parameters:,}")

        @staticmethod
        def scheduled_sparsity(step: int) -> float:
            progress = min(max(step / max(1, end_step), 0.0), 1.0)
            return final_sparsity + (initial_sparsity - final_sparsity) * (1.0 - progress) ** 3

        def update_masks(self, target_sparsity: float) -> None:
            absolute_values = [
                np.abs(np.asarray(weight.numpy())) for weight, _ in self.weight_masks
            ]
            magnitudes = np.concatenate([values.reshape(-1) for values in absolute_values])
            prune_count = int(magnitudes.size * target_sparsity)
            threshold = (
                float(np.partition(magnitudes, prune_count - 1)[prune_count - 1])
                if prune_count > 0
                else -1.0
            )

            kept = 0
            for values, (_, mask) in zip(absolute_values, self.weight_masks):
                keep = values > threshold
                mask.assign(keep)
                kept += int(np.count_nonzero(keep))
            self.parameter_count = int(magnitudes.size)
            self.zero_count = self.parameter_count - kept
            self.last_sparsity = target_sparsity
            self.actual_sparsity = self.zero_count / self.parameter_count

        def apply_masks(self) -> None:
            self._apply_compiled()

        def on_train_batch_end(
            self,
            batch: int,
            logs: Optional[Dict[str, Any]] = None,
        ) -> None:
            step = int(self.model.optimizer.iterations.numpy())
            self.apply_masks()
            should_update = step < end_step and step % frequency == 0
            should_finish = step >= end_step and self.last_sparsity < final_sparsity
            if should_update or should_finish:
                self.update_masks(self.scheduled_sparsity(step))
                self.apply_masks()

        def on_epoch_end(
            self,
            epoch: int,
            logs: Optional[Dict[str, Any]] = None,
        ) -> None:
            print(
                f"pruning sparsity after epoch {epoch + 1}: "
                f"{self.actual_sparsity:.2%} (scheduled {self.last_sparsity:.2%})"
            )

        def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
            if self.last_sparsity < final_sparsity:
                self.update_masks(final_sparsity)
                self.apply_masks()

    return MagnitudePruningCallback()


def validate_model_dataset(model: Any, vocab_sizes: Dict[str, int]) -> int:
    input_names = {tensor.name.split(":", 1)[0].split("/")[-1] for tensor in model.inputs}
    if input_names != set(INPUT_FIELDS):
        raise ValueError(f"model inputs do not match dataset fields: {sorted(input_names)}")

    sequence_lengths = {tensor.shape[1] for tensor in model.inputs}
    if len(sequence_lengths) != 1 or None in sequence_lengths:
        raise ValueError(f"model inputs must have one fixed sequence length: {sequence_lengths}")

    output_sizes = {
        name: int(tensor.shape[-1])
        for name, tensor in zip(model.output_names, model.outputs)
    }
    expected = {
        "next_dst_node": int(vocab_sizes["node"]),
        "next_ctrl_type": int(vocab_sizes["ctrl_type"]),
    }
    if output_sizes != expected:
        raise ValueError(
            f"model outputs {output_sizes} do not match dataset vocabularies {expected}"
        )
    return int(next(iter(sequence_lengths)))


def json_values(values: Dict[str, Any]) -> Dict[str, float]:
    return {name: float(value) for name, value in values.items()}


def compress(cfg: SimpleNamespace) -> Dict[str, Any]:
    configure_gpu_memory_growth()
    tf.keras.utils.set_random_seed(cfg.seed)

    dataset_dir = cfg.dataset_dir.expanduser().resolve()
    input_model = cfg.input_model.expanduser().resolve()
    output_model = cfg.output_model.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)

    vocab_sizes = load_vocab_sizes(dataset_dir)
    model = load_flow_model(input_model)
    max_seq_len = validate_model_dataset(model, vocab_sizes)

    train_data = FlowSequence(
        dataset_dir,
        max_seq_len=max_seq_len,
        batch_size=cfg.batch_size,
        mode="train",
        validation_split=cfg.validation_split,
        shuffle=True,
        seed=cfg.seed,
    )
    val_data = FlowSequence(
        dataset_dir,
        max_seq_len=max_seq_len,
        batch_size=cfg.batch_size,
        mode="val",
        validation_split=cfg.validation_split,
        shuffle=False,
        seed=cfg.seed,
    )
    if len(train_data) == 0:
        raise ValueError("no training batches are available")

    compile_model(model, cfg, vocab_sizes)

    evaluation_data = val_data if len(val_data) else train_data
    baseline_metrics = json_values(model.evaluate(evaluation_data, verbose=1, return_dict=True))
    prunable = selected_weights(model, cfg.exclude_patterns)
    baseline_sparsity = sparsity_stats(prunable)

    pruning_end_step = len(train_data) * (cfg.epochs - cfg.recovery_epochs)
    pruning_frequency = max(1, (pruning_end_step + cfg.pruning_updates - 1) // cfg.pruning_updates)
    pruning = build_pruning_callback(
        initial_sparsity=cfg.initial_sparsity,
        final_sparsity=cfg.final_sparsity,
        end_step=pruning_end_step,
        frequency=pruning_frequency,
        exclude_patterns=cfg.exclude_patterns,
    )
    history = model.fit(
        train_data,
        validation_data=val_data if len(val_data) else None,
        epochs=cfg.epochs,
        callbacks=[pruning],
    )
    compressed_metrics = json_values(model.evaluate(evaluation_data, verbose=1, return_dict=True))
    compressed_sparsity = {
        "tensors": len(pruning.weight_masks),
        "parameters": pruning.parameter_count,
        "zero_parameters": pruning.zero_count,
        "sparsity": pruning.actual_sparsity,
    }

    history_path = output_model.with_suffix(output_model.suffix + ".history.json")
    model.save(output_model)
    with history_path.open("w", encoding="utf-8") as stream:
        json.dump(history.history, stream, ensure_ascii=False, indent=2)

    input_size = input_model.stat().st_size
    keras_size = output_model.stat().st_size

    report = {
        "config_file": str(cfg.config_file),
        "input_model": str(input_model),
        "output_model": str(output_model),
        "max_seq_len": max_seq_len,
        "settings": {
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "recovery_epochs": cfg.recovery_epochs,
            "learning_rate": cfg.learning_rate,
            "initial_sparsity": cfg.initial_sparsity,
            "final_sparsity": cfg.final_sparsity,
            "pruning_updates": cfg.pruning_updates,
        },
        "baseline_metrics": baseline_metrics,
        "compressed_metrics": compressed_metrics,
        "baseline_sparsity": baseline_sparsity,
        "compressed_sparsity": compressed_sparsity,
        "file_sizes": {
            "input_keras": input_size,
            "pruned_keras": keras_size,
        },
        "size_reduction": {
            "pruned_keras": 1.0 - keras_size / input_size,
        },
    }
    report_path = output_model.with_suffix(output_model.suffix + ".compression.json")
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    report["report"] = str(report_path)
    report["history"] = str(history_path)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune and fine-tune a flow model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        validate_config(cfg)
        report = compress(cfg)
    except (ImportError, KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    sizes = report["file_sizes"]
    print("pruning and fine-tuning complete")
    print(f"pruned sparsity: {report['compressed_sparsity']['sparsity']:.2%}")
    print(f"input keras: {sizes['input_keras'] / 1024**2:.2f} MiB")
    print(f"pruned keras: {sizes['pruned_keras'] / 1024**2:.2f} MiB")
    print(f"keras model: {report['output_model']}")
    print(f"report: {report['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
