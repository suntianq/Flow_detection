"""Plain unidirectional LSTM baseline for structured flow-token prediction."""

from __future__ import annotations

from typing import Any, Dict

from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, LSTM

from .common import build_token_encoder, cfg_value, parse_units, prediction_heads


def build_lstm_model(cfg: Any, vocab_sizes: Dict[str, int]) -> Model:
    inputs, x, sizes = build_token_encoder(cfg, vocab_sizes, mask_zero=True)
    dropout = float(cfg_value(cfg, "dropout", 0.3))
    units_list = parse_units(cfg_value(cfg, "lstm_units", None), [128, 64])

    for layer_idx, units in enumerate(units_list):
        return_sequences = layer_idx < len(units_list) - 1
        x = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout,
            name=f"lstm_{layer_idx + 1}",
        )(x)

    x = Dropout(dropout, name="head_dropout")(x)
    return Model(inputs=inputs, outputs=prediction_heads(x, sizes), name="flow_lstm")
