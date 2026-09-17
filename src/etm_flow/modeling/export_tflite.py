#!/usr/bin/env python3
"""Export a Keras model as a portable TFLite model.

The exporter deliberately hides GPUs before importing TensorFlow. This prevents
Keras LSTM/GRU layers configured with ``use_cudnn="auto"`` from being traced as
GPU-specific CudnnRNN ops. Training can still use cuDNN; only this standalone
export process is CPU-traced.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# TensorFlow reads this variable while it is imported. It must be set here,
# before the third-party import block, so the exported RNN graph cannot capture
# CudnnRNNV3. All imports remain at module scope.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import yaml
from tensorflow.keras.models import load_model

from etm_flow.data.dataset import FlowSequence, load_vocab_sizes
from etm_flow.modeling.models.common import LearnedPositionEmbedding


def log_status(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@contextmanager
def monitored_stage(label: str, heartbeat_seconds: float) -> Iterator[None]:
    started = time.perf_counter()
    stopped = threading.Event()

    def report_heartbeat() -> None:
        while not stopped.wait(heartbeat_seconds):
            elapsed = time.perf_counter() - started
            log_status(f"{label} still running ({elapsed:.1f}s elapsed)")

    log_status(f"{label}...")
    thread = threading.Thread(target=report_heartbeat, daemon=True)
    thread.start()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        stopped.set()
        thread.join()
        elapsed = time.perf_counter() - started
        outcome = "complete" if succeeded else "failed"
        log_status(f"{label} {outcome} ({elapsed:.1f}s)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Keras model with a portable, CPU-traced TFLite graph."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--input-model",
        type=Path,
        help="Keras model to export; defaults to compress.output_model.",
    )
    parser.add_argument(
        "--output-tflite",
        type=Path,
        help="Destination; defaults to compress.output_tflite.",
    )
    parser.add_argument(
        "--quantization",
        choices=("dynamic", "float16", "int8"),
        help="Override compress.quantization.",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        help="Override compress.representative_samples for INT8 calibration.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help=(
            "Fixed batch dimension in the exported graph (default: 1). "
            "A static batch lets TFLite lower LSTM/GRU TensorList state."
        ),
    )
    parser.add_argument(
        "--integer-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require TFLITE_BUILTINS_INT8 (when quantization is int8).",
    )
    parser.add_argument(
        "--allow-select-tf-ops",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow Flex/Select TF Ops. This reduces deployment portability.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress heartbeats for long stages (default: 10).",
    )
    parser.add_argument(
        "--force-int8-rnn",
        action="store_true",
        help=(
            "Attempt full INT8 calibration for an unfused LSTM/GRU graph. "
            "This is unsafe with some TensorFlow versions and may segfault."
        ),
    )
    parser.add_argument(
        "--sparsity-optimization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Encode eligible zero weights with experimental TFLite sparsity. "
            "Defaults to off for dynamic quantization so XNNPACK can execute "
            "the fully-connected operators."
        ),
    )
    return parser


def load_config(args: argparse.Namespace) -> SimpleNamespace:
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    compress = raw["compress"]
    dataset = raw["dataset"]
    training = raw["training"]
    input_model = (
        args.input_model
        if args.input_model is not None
        else Path(compress["output_model"])
    )
    output_tflite = (
        args.output_tflite
        if args.output_tflite is not None
        else Path(compress["output_tflite"])
    )
    quantization = (
        args.quantization
        if args.quantization is not None
        else str(compress["quantization"]).lower()
    )
    representative_samples = (
        args.representative_samples
        if args.representative_samples is not None
        else int(compress["representative_samples"])
    )
    integer_only = (
        args.integer_only
        if args.integer_only is not None
        else bool(compress["integer_only"])
    )
    allow_select_tf_ops = (
        args.allow_select_tf_ops
        if args.allow_select_tf_ops is not None
        else bool(compress["allow_select_tf_ops"])
    )
    sparsity_optimization = (
        args.sparsity_optimization
        if args.sparsity_optimization is not None
        else quantization != "dynamic"
    )
    return SimpleNamespace(
        config_file=config_path,
        dataset_dir=Path(dataset["dir"]).expanduser().resolve(),
        input_model=input_model.expanduser().resolve(),
        output_tflite=output_tflite.expanduser().resolve(),
        batch_size=int(compress["batch_size"]),
        validation_split=float(training["validation_split"]),
        seed=int(training["seed"]),
        quantization=quantization,
        representative_samples=representative_samples,
        integer_only=integer_only,
        allow_select_tf_ops=allow_select_tf_ops,
        inference_batch_size=int(args.inference_batch_size),
        progress_interval=float(args.progress_interval),
        force_int8_rnn=bool(args.force_int8_rnn),
        sparsity_optimization=bool(sparsity_optimization),
    )


def validate_config(cfg: SimpleNamespace) -> None:
    if not cfg.input_model.is_file():
        raise ValueError(
            f"input model not found: {cfg.input_model}. "
            "compress.py saves compress.output_model before TFLite conversion; "
            "use --input-model to select another .keras file if needed."
        )
    if cfg.input_model.suffix != ".keras":
        raise ValueError("input_model must end with .keras")
    if cfg.output_tflite.suffix != ".tflite":
        raise ValueError("output_tflite must end with .tflite")
    if cfg.quantization not in {"dynamic", "float16", "int8"}:
        raise ValueError("quantization must be dynamic, float16, or int8")
    if cfg.quantization == "int8":
        if not cfg.dataset_dir.is_dir():
            raise ValueError(
                f"dataset directory not found for INT8 calibration: {cfg.dataset_dir}"
            )
        if cfg.representative_samples <= 0:
            raise ValueError("representative_samples must be > 0 for int8")
    if cfg.integer_only and cfg.quantization != "int8":
        raise ValueError("integer_only is only valid with int8 quantization")
    if cfg.integer_only and cfg.allow_select_tf_ops:
        raise ValueError("integer_only and allow_select_tf_ops cannot both be true")
    if cfg.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")
    if (
        cfg.quantization == "int8"
        and cfg.inference_batch_size > cfg.batch_size
    ):
        raise ValueError(
            "inference_batch_size cannot exceed compress.batch_size during "
            "INT8 calibration"
        )
    if cfg.progress_interval <= 0:
        raise ValueError("progress_interval must be > 0")


def inference_signature(
    model: Any,
    batch_size: int,
) -> Dict[str, Any]:
    signature: Dict[str, Any] = {}
    for tensor in model.inputs:
        name = tensor.name.split(":", 1)[0].split("/")[-1]
        sample_shape = [
            int(dimension) if dimension is not None else None
            for dimension in tensor.shape[1:]
        ]
        signature[name] = tf.TensorSpec(
            shape=[batch_size, *sample_shape],
            dtype=tensor.dtype,
            name=name,
        )
    return signature


def graph_op_types(concrete_function: Any) -> Sequence[str]:
    graph_def = concrete_function.graph.as_graph_def()
    op_types = {node.op for node in graph_def.node}
    for function in graph_def.library.function:
        op_types.update(node.op for node in function.node_def)
    return sorted(op_types)


def recurrent_layer_names(model: Any) -> Sequence[str]:
    recurrent_types = (tf.keras.layers.LSTM, tf.keras.layers.GRU)
    names = []
    pending = list(getattr(model, "layers", ()))
    visited = set()
    while pending:
        layer = pending.pop()
        identity = id(layer)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(layer, recurrent_types):
            names.append(layer.name)
        pending.extend(getattr(layer, "layers", ()))
    return sorted(names)


def saved_model_op_types(saved_model_dir: str) -> Sequence[str]:
    exported = tf.saved_model.load(saved_model_dir)
    op_types = set()
    for concrete_function in exported.signatures.values():
        op_types.update(graph_op_types(concrete_function))
    return sorted(op_types)


def collect_representative_batches(
    train_data: Any,
    sample_limit: int,
    inference_batch_size: int,
) -> Tuple[List[Dict[str, Any]], int]:
    batches: List[Dict[str, Any]] = []
    emitted = 0
    report_every = max(inference_batch_size, min(64, sample_limit))
    next_report = report_every
    log_status(
        "Collecting INT8 calibration samples "
        f"(target={sample_limit}, batch={inference_batch_size})..."
    )
    for batch_index in range(len(train_data)):
        inputs = train_data[batch_index][0]
        batch_size = len(next(iter(inputs.values())))
        for row in range(0, batch_size, inference_batch_size):
            stop = row + inference_batch_size
            if stop > batch_size:
                break
            batches.append({
                name: values[row:stop].copy()
                for name, values in inputs.items()
            })
            emitted += inference_batch_size
            if emitted >= next_report or emitted >= sample_limit:
                log_status(
                    "INT8 calibration samples ready: "
                    f"{min(emitted, sample_limit)}/{sample_limit}"
                )
                next_report += report_every
            if emitted >= sample_limit:
                return batches, emitted
    return batches, emitted


def representative_dataset(
    batches: Sequence[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    yield from batches


def tflite_op_names(interpreter: Any) -> Sequence[str]:
    get_ops = getattr(interpreter, "_get_ops_details", None)
    if get_ops is None:
        return []
    return sorted({str(item["op_name"]) for item in get_ops()})


def convert(
    model: Any,
    calibration_batches: Optional[Sequence[Dict[str, Any]]],
    cfg: SimpleNamespace,
) -> tuple[bytes, Dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="portable_savedmodel_") as export_dir:
        # Keras 3 tracks Dropout SeedGenerator state and other Keras-specific
        # resources through Model.export(). Calling tf.saved_model.save() on a
        # hand-written closure can leave those resources untracked.
        with monitored_stage(
            "Exporting CPU-traced SavedModel",
            cfg.progress_interval,
        ):
            model.export(
                export_dir,
                input_signature=[
                    inference_signature(model, cfg.inference_batch_size)
                ],
            )

        with monitored_stage(
            "Inspecting SavedModel operators",
            cfg.progress_interval,
        ):
            source_ops = saved_model_op_types(export_dir)
        gpu_specific_ops = [
            name
            for name in source_ops
            if "Cudnn" in name or "Rocm" in name or "Mkl" in name
        ]
        if gpu_specific_ops:
            raise RuntimeError(
                "portable export still contains hardware-specific ops: "
                f"{gpu_specific_ops}. Run this file as a fresh Python process and "
                "do not import TensorFlow before it starts."
            )

        rnn_layers = recurrent_layer_names(model)
        tensor_list_ops = [name for name in source_ops if "TensorList" in name]
        if (
            cfg.quantization == "int8"
            and rnn_layers
            and tensor_list_ops
            and not cfg.force_int8_rnn
        ):
            raise RuntimeError(
                "full INT8 conversion was stopped before TensorFlow calibration "
                "because this Keras RNN is not represented as a fused TFLite LSTM. "
                f"RNN layers: {rnn_layers}; TensorList ops: {tensor_list_ops}. "
                "This combination segfaulted in the current TensorFlow converter. "
                "Use '--quantization dynamic' for a portable CPU model, or "
                "'--quantization float16' for a floating-point deployment model. "
                "Use --force-int8-rnn only to retest after changing TensorFlow, "
                "because a native crash cannot be caught by Python."
            )

        converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if cfg.sparsity_optimization:
            converter.optimizations.append(
                tf.lite.Optimize.EXPERIMENTAL_SPARSITY
            )

        if cfg.quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif cfg.quantization == "int8":
            if not calibration_batches:
                raise RuntimeError("INT8 conversion requires representative data")
            converter.representative_dataset = lambda: representative_dataset(
                calibration_batches,
            )
            if cfg.integer_only:
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]

        if cfg.allow_select_tf_ops:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_lower_tensor_list_ops = False

        try:
            with monitored_stage(
                f"Converting to TFLite ({cfg.quantization})",
                cfg.progress_interval,
            ):
                tflite_model = converter.convert()
        except Exception as exc:
            message = str(exc)
            if "TensorList" in message:
                hint = (
                    "The graph was CPU-traced, so this is not a cuDNN problem. "
                    "A fixed --inference-batch-size is already used to make RNN "
                    "state shapes static. If TensorList lowering still fails, the "
                    "current TensorFlow converter cannot lower this RNN variant; "
                    "--allow-select-tf-ops is the compatibility fallback, but it "
                    "produces a less-portable Flex model."
                )
            else:
                hint = (
                    "The graph was CPU-traced, so this is not a cuDNN export "
                    "problem. Try --quantization dynamic first. GRU and some "
                    "Transformer graphs may require --allow-select-tf-ops, but "
                    "that produces a Flex model and is less portable."
                )
            raise RuntimeError(
                f"TFLite conversion failed: {type(exc).__name__}: {message}\n{hint}"
            ) from exc

    with monitored_stage("Validating TFLite model", cfg.progress_interval):
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        converted_ops = tflite_op_names(interpreter)
    flex_ops = [name for name in converted_ops if name.startswith("Flex")]
    if flex_ops and not cfg.allow_select_tf_ops:
        raise RuntimeError(f"unexpected Flex ops in portable model: {flex_ops}")

    details = {
        "source_ops": source_ops,
        "tflite_ops": converted_ops,
        "flex_ops": flex_ops,
        "inputs": [item["name"] for item in interpreter.get_input_details()],
        "outputs": [item["name"] for item in interpreter.get_output_details()],
        "signatures": interpreter.get_signature_list(),
    }
    return tflite_model, details


def main(argv: Optional[Sequence[str]] = None) -> int:
    total_started = time.perf_counter()
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(args)
        validate_config(cfg)
        tf.config.set_visible_devices([], "GPU")
        tf.keras.utils.set_random_seed(cfg.seed)
        log_status(
            "Portable export started: "
            f"quantization={cfg.quantization}, "
            f"batch={cfg.inference_batch_size}, "
            f"sparsity_optimization={cfg.sparsity_optimization}, "
            f"input={cfg.input_model}"
        )

        custom_objects = {
            "LearnedPositionEmbedding": LearnedPositionEmbedding,
            "FlowModels>LearnedPositionEmbedding": LearnedPositionEmbedding,
        }
        with monitored_stage("Loading Keras model", cfg.progress_interval):
            model = load_model(
                str(cfg.input_model),
                custom_objects=custom_objects,
                compile=False,
            )

        train_data = None
        calibration_batches = None
        calibration_sample_count = 0
        max_seq_len = None
        if cfg.quantization == "int8":
            with monitored_stage(
                "Opening calibration dataset",
                cfg.progress_interval,
            ):
                load_vocab_sizes(cfg.dataset_dir)
                sequence_lengths = {tensor.shape[1] for tensor in model.inputs}
                if len(sequence_lengths) != 1 or None in sequence_lengths:
                    raise ValueError(
                        "model inputs must share one fixed sequence length "
                        "for calibration"
                    )
                max_seq_len = int(next(iter(sequence_lengths)))
                train_data = FlowSequence(
                    cfg.dataset_dir,
                    max_seq_len=max_seq_len,
                    batch_size=cfg.batch_size,
                    mode="train",
                    validation_split=cfg.validation_split,
                    shuffle=False,
                    seed=cfg.seed,
                )
                if len(train_data) == 0:
                    raise ValueError(
                        "no training batches are available for calibration"
                    )
            calibration_batches, calibration_sample_count = (
                collect_representative_batches(
                    train_data,
                    cfg.representative_samples,
                    cfg.inference_batch_size,
                )
            )
            if not calibration_batches:
                raise ValueError("no complete calibration batches are available")

        tflite_model, details = convert(model, calibration_batches, cfg)
        with monitored_stage("Writing TFLite model", cfg.progress_interval):
            cfg.output_tflite.parent.mkdir(parents=True, exist_ok=True)
            cfg.output_tflite.write_bytes(tflite_model)

        report = {
            "config_file": str(cfg.config_file),
            "input_model": str(cfg.input_model),
            "output_tflite": str(cfg.output_tflite),
            "quantization": cfg.quantization,
            "integer_only": cfg.integer_only,
            "allow_select_tf_ops": cfg.allow_select_tf_ops,
            "force_int8_rnn": cfg.force_int8_rnn,
            "sparsity_optimization": cfg.sparsity_optimization,
            "inference_batch_size": cfg.inference_batch_size,
            "representative_samples": (
                calibration_sample_count if cfg.quantization == "int8" else 0
            ),
            "max_seq_len": max_seq_len,
            "input_bytes": cfg.input_model.stat().st_size,
            "output_bytes": len(tflite_model),
            "size_reduction": 1.0 - len(tflite_model) / cfg.input_model.stat().st_size,
            **details,
        }
        report_path = cfg.output_tflite.with_suffix(
            cfg.output_tflite.suffix + ".export.json"
        )
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)

        elapsed = time.perf_counter() - total_started
        log_status(f"Portable TFLite export complete ({elapsed:.1f}s total)")
        print(f"input model: {cfg.input_model}")
        print(f"output model: {cfg.output_tflite}")
        print(f"quantization: {cfg.quantization}")
        print(f"Flex ops: {details['flex_ops'] or 'none'}")
        print(f"size: {len(tflite_model) / 1024**2:.2f} MiB")
        print(f"report: {report_path}")
        return 0
    except (ImportError, KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"etm-export-tflite: error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
