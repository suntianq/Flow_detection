"""CNN + LSTM baseline for structured flow-token prediction."""

from __future__ import annotations

from typing import Any, Dict

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv1D,
    Dropout,
    LSTM,
    LayerNormalization,
    MaxPooling1D,
)

from .common import build_token_encoder, cfg_value, parse_units, prediction_heads


def build_cnn_lstm_model(cfg: Any, vocab_sizes: Dict[str, int]) -> Model:
    inputs, x, sizes = build_token_encoder(cfg, vocab_sizes)
    dropout = float(cfg_value(cfg, "dropout", 0.3))

    conv_blocks = int(cfg_value(cfg, "conv_blocks", 2))
    conv_filters = int(cfg_value(cfg, "conv_filters", 128))
    conv_kernel = int(cfg_value(cfg, "conv_kernel", 5))
    pool_size = int(cfg_value(cfg, "pool_size", 2))
    for block_idx in range(conv_blocks):
        x = Conv1D(
            conv_filters,
            kernel_size=conv_kernel,
            padding="same",
            activation="relu",
            name=f"conv_{block_idx + 1}",
        )(x)
        if pool_size > 1:
            x = MaxPooling1D(pool_size=pool_size, name=f"pool_{block_idx + 1}")(x)
        x = LayerNormalization(name=f"conv_norm_{block_idx + 1}")(x)

    lstm_units = parse_units(cfg_value(cfg, "lstm_units", None), [128, 64])
    for layer_idx, units in enumerate(lstm_units):
        return_sequences = layer_idx < len(lstm_units) - 1
        x = LSTM(units, return_sequences=return_sequences, dropout=dropout, name=f"lstm_{layer_idx + 1}")(x)

    x = Dropout(dropout, name="head_dropout")(x)
    return Model(inputs=inputs, outputs=prediction_heads(x, sizes), name="flow_cnn_lstm")
