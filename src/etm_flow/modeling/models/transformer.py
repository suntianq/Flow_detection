"""Small Transformer encoder model for structured flow-token prediction."""

from __future__ import annotations

from typing import Any, Dict

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Cropping1D,
    Dense,
    Dropout,
    Flatten,
    LayerNormalization,
    MultiHeadAttention,
)

from .common import build_token_encoder, cfg_value, prediction_heads


def transformer_block(
    x: Any,
    *,
    model_dim: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    block_idx: int,
) -> Any:
    key_dim = max(1, model_dim // num_heads)
    attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=f"transformer_{block_idx}_mha",
    )(x, x)
    attention = Dropout(dropout, name=f"transformer_{block_idx}_attn_dropout")(attention)
    x = Add(name=f"transformer_{block_idx}_attn_add")([x, attention])
    x = LayerNormalization(name=f"transformer_{block_idx}_attn_norm")(x)

    ffn = Dense(ff_dim, activation="relu", name=f"transformer_{block_idx}_ffn_1")(x)
    ffn = Dropout(dropout, name=f"transformer_{block_idx}_ffn_dropout")(ffn)
    ffn = Dense(model_dim, name=f"transformer_{block_idx}_ffn_2")(ffn)
    x = Add(name=f"transformer_{block_idx}_ffn_add")([x, ffn])
    return LayerNormalization(name=f"transformer_{block_idx}_ffn_norm")(x)


def build_transformer_model(cfg: Any, vocab_sizes: Dict[str, int]) -> Model:
    inputs, x, sizes = build_token_encoder(cfg, vocab_sizes)
    max_seq_len = int(cfg_value(cfg, "max_seq_len", 64))
    model_dim = int(cfg_value(cfg, "model_dim", 128))
    dropout = float(cfg_value(cfg, "dropout", 0.3))
    layers = int(cfg_value(cfg, "transformer_layers", 2))
    heads = int(cfg_value(cfg, "transformer_heads", 4))
    ff_dim = int(cfg_value(cfg, "transformer_ff_dim", model_dim * 2))

    if heads <= 0:
        raise ValueError("transformer_heads must be > 0")

    for block_idx in range(1, layers + 1):
        x = transformer_block(
            x,
            model_dim=model_dim,
            num_heads=heads,
            ff_dim=ff_dim,
            dropout=dropout,
            block_idx=block_idx,
        )

    x = Cropping1D(cropping=(max_seq_len - 1, 0), name="last_timestep_crop")(x)
    x = Flatten(name="last_timestep")(x)
    x = Dropout(dropout, name="head_dropout")(x)
    return Model(inputs=inputs, outputs=prediction_heads(x, sizes), name="flow_transformer")
