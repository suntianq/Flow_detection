"""Dilated causal CNN / TCN model for structured flow-token prediction."""

from __future__ import annotations

from typing import Any, Dict, List

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv1D,
    Cropping1D,
    Dropout,
    Flatten,
    LayerNormalization,
)

from .common import build_token_encoder, cfg_value, parse_units, prediction_heads


def parse_dilations(value: Any) -> List[int]:
    return parse_units(value, [1, 2, 4, 8])


def residual_tcn_block(x: Any, *, filters: int, kernel_size: int, dilation: int, dropout: float, block_idx: int) -> Any:
    residual = x
    y = Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="causal",
        name=f"tcn_{block_idx}_conv_1",
    )(x)
    y = LayerNormalization(name=f"tcn_{block_idx}_norm_1")(y)
    y = Activation("relu", name=f"tcn_{block_idx}_relu_1")(y)
    y = Dropout(dropout, name=f"tcn_{block_idx}_dropout_1")(y)

    y = Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="causal",
        name=f"tcn_{block_idx}_conv_2",
    )(y)
    y = LayerNormalization(name=f"tcn_{block_idx}_norm_2")(y)

    if residual.shape[-1] != filters:
        residual = Conv1D(filters, kernel_size=1, padding="same", name=f"tcn_{block_idx}_residual")(residual)

    y = Add(name=f"tcn_{block_idx}_add")([residual, y])
    return Activation("relu", name=f"tcn_{block_idx}_relu_out")(y)


def build_tcn_model(cfg: Any, vocab_sizes: Dict[str, int]) -> Model:
    inputs, x, sizes = build_token_encoder(cfg, vocab_sizes)
    max_seq_len = int(cfg_value(cfg, "max_seq_len", 64))
    dropout = float(cfg_value(cfg, "dropout", 0.3))
    filters = int(cfg_value(cfg, "tcn_filters", cfg_value(cfg, "conv_filters", 128)))
    kernel_size = int(cfg_value(cfg, "tcn_kernel", cfg_value(cfg, "conv_kernel", 5)))
    dilations = parse_dilations(cfg_value(cfg, "tcn_dilations", None))

    for block_idx, dilation in enumerate(dilations, start=1):
        x = residual_tcn_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation=int(dilation),
            dropout=dropout,
            block_idx=block_idx,
        )

    x = Cropping1D(cropping=(max_seq_len - 1, 0), name="last_timestep_crop")(x)
    x = Flatten(name="last_timestep")(x)
    x = Dropout(dropout, name="head_dropout")(x)
    return Model(inputs=inputs, outputs=prediction_heads(x, sizes), name="flow_tcn")
