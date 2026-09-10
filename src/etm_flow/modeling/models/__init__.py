"""Model factory for flow-sequence training."""

from __future__ import annotations

from typing import Any, Dict

from .cnn_lstm import build_cnn_lstm_model
from .gru import build_gru_model
from .lstm import build_lstm_model
from .tcn import build_tcn_model
from .transformer import build_transformer_model


def build_model(model_name: str, cfg: Any, vocab_sizes: Dict[str, int]):
    normalized = model_name.strip().lower().replace("-", "_")
    if normalized in {"cnn_lstm", "cnn+lstm"}:
        return build_cnn_lstm_model(cfg, vocab_sizes)
    if normalized == "lstm":
        return build_lstm_model(cfg, vocab_sizes)
    if normalized == "gru":
        return build_gru_model(cfg, vocab_sizes)
    if normalized in {"tcn", "causal_cnn", "dilated_cnn"}:
        return build_tcn_model(cfg, vocab_sizes)
    if normalized in {"transformer", "small_transformer"}:
        return build_transformer_model(cfg, vocab_sizes)
    raise ValueError(f"unsupported model: {model_name}")
