#!/usr/bin/env python3
"""Train flow-sequence next-token models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers import Adam

from etm_flow.data.dataset import FlowSequence, load_vocab_sizes
from etm_flow.modeling.models import build_model


def configure_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def build_loss_weights(cfg: SimpleNamespace) -> Dict[str, float]:
    return {
        "next_dst_node": float(cfg.loss_weight_dst_node),
        "next_ctrl_type": float(cfg.loss_weight_ctrl_type),
    }


def compile_model(model: Any, cfg: SimpleNamespace, vocab_sizes: Dict[str, int]) -> None:
    losses = {
        "next_dst_node": "sparse_categorical_crossentropy",
        "next_ctrl_type": "sparse_categorical_crossentropy",
    }
    metrics = {
        "next_dst_node": ["accuracy"],
        "next_ctrl_type": ["accuracy"],
    }
    model.compile(
        optimizer=Adam(learning_rate=float(cfg.learning_rate)),
        loss=losses,
        loss_weights=build_loss_weights(cfg),
        metrics=metrics,
    )


def load_config(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    dataset = cfg["dataset"]
    model = cfg["model"]
    training = cfg["training"]
    loss_weights = cfg["loss_weights"]

    values = {
        "dataset_dir": Path(dataset["dir"]),
        "model_name": model["name"],
        **{key: value for key, value in model.items() if key != "name"},
        **training,
        **{f"loss_weight_{key}": value for key, value in loss_weights.items()},
    }
    values["output_model"] = Path(training["output_model"])
    return SimpleNamespace(**values)


def validate_config(cfg: SimpleNamespace) -> None:
    def required(name: str) -> Any:
        if not hasattr(cfg, name):
            raise ValueError(f"config is missing required field: {name}")
        return getattr(cfg, name)

    if int(required("max_seq_len")) <= 0:
        raise ValueError("max_seq_len must be > 0")
    if int(required("batch_size")) <= 0:
        raise ValueError("batch_size must be > 0")
    if not 0.0 <= float(required("validation_split")) < 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0)")
    if int(required("epochs")) <= 0:
        raise ValueError("epochs must be > 0")
    if float(required("learning_rate")) <= 0:
        raise ValueError("learning_rate must be > 0")
    if int(required("seed")) < 0:
        raise ValueError("seed must be >= 0")
    if int(required("patience")) < 0:
        raise ValueError("patience must be >= 0")
    if int(required("lr_patience")) < 0:
        raise ValueError("lr_patience must be >= 0")
    float(required("loss_weight_dst_node"))
    float(required("loss_weight_ctrl_type"))
    required("model_name")
    required("dataset_dir")
    required("output_model")


def write_training_config(cfg: SimpleNamespace, input_config_path: Path, vocab_sizes: Dict[str, int]) -> None:
    output_path = cfg.output_model.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_path.with_suffix(output_path.suffix + ".config.json")
    record = vars(cfg).copy()
    record["config_file"] = str(input_config_path)
    record["dataset_dir"] = str(cfg.dataset_dir)
    record["output_model"] = str(cfg.output_model)
    record["vocab_sizes"] = vocab_sizes
    with resolved_config_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a structured flow next-token model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Training config YAML file. Default: configs/default.yaml",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    cli_args = parser.parse_args(argv)

    try:
        cfg = load_config(cli_args.config)
        validate_config(cfg)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    if getattr(cfg, "mixed_precision", False):
        set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")

    configure_gpu_memory_growth()

    dataset_dir = cfg.dataset_dir.expanduser().resolve()
    vocab_sizes = load_vocab_sizes(dataset_dir)
    print(f"vocab_sizes: {vocab_sizes}")

    train_gen = FlowSequence(
        dataset_dir,
        max_seq_len=int(cfg.max_seq_len),
        batch_size=int(cfg.batch_size),
        mode="train",
        validation_split=float(cfg.validation_split),
        shuffle=True,
        seed=int(cfg.seed),
    )
    val_gen = FlowSequence(
        dataset_dir,
        max_seq_len=int(cfg.max_seq_len),
        batch_size=int(cfg.batch_size),
        mode="val",
        validation_split=float(cfg.validation_split),
        shuffle=False,
        seed=int(cfg.seed),
    )
    if len(train_gen) == 0:
        parser.error("no train batches available; reduce --batch-size or check dataset size")

    model = build_model(str(cfg.model_name), cfg, vocab_sizes)
    compile_model(model, cfg, vocab_sizes)
    model.summary()

    monitor = "val_loss" if len(val_gen) > 0 else "loss"
    output_model = cfg.output_model.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=int(cfg.lr_patience), min_lr=1e-5, verbose=1),
        EarlyStopping(monitor=monitor, patience=int(cfg.patience), restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(output_model), save_best_only=True, monitor=monitor, verbose=1),
    ]

    write_training_config(cfg, cli_args.config, vocab_sizes)
    print(f"train_batches: {len(train_gen)}")
    print(f"val_batches: {len(val_gen)}")
    history = model.fit(
        train_gen,
        validation_data=val_gen if len(val_gen) > 0 else None,
        epochs=int(cfg.epochs),
        callbacks=callbacks,
    )

    history_path = output_model.with_suffix(output_model.suffix + ".history.json")
    with history_path.open("w", encoding="utf-8") as stream:
        json.dump(history.history, stream, ensure_ascii=False, indent=2)

    print(f"training complete: {output_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
