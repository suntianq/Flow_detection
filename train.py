#!/usr/bin/env python3
"""Train flow-sequence next-token models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence


def configure_gpu_memory_growth() -> None:
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def build_loss_weights(cfg: SimpleNamespace) -> Dict[str, float]:
    return {
        "next_dst_node": float(cfg.loss_weight_dst_node),
        "next_ctrl_type": float(cfg.loss_weight_ctrl_type),
    }


def compile_model(model: Any, cfg: SimpleNamespace, vocab_sizes: Dict[str, int]) -> None:
    from tensorflow.keras.optimizers import Adam

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
    import yaml

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
    if int(cfg.max_seq_len) <= 0:
        raise ValueError("max_seq_len must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if not 0.0 <= float(cfg.validation_split) < 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0)")


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
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Training config YAML file.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    cli_args = parser.parse_args(argv)

    try:
        cfg = load_config(cli_args.config)
        validate_config(cfg)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if cfg.mixed_precision:
        from tensorflow.keras.mixed_precision import set_global_policy

        set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")

    configure_gpu_memory_growth()

    from dataset_builder import FlowSequence, load_vocab_sizes
    from models import build_model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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
