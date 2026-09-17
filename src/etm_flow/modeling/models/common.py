"""Shared building blocks for structured flow-token models."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Activation, Concatenate, Dense, Embedding, Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable

from etm_flow.data.dataset import INPUT_FIELDS


@register_keras_serializable(package="FlowModels")
class LearnedPositionEmbedding(Layer):
    def __init__(self, max_len: int, embed_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 加位置向量是逐元素操作，保持时间维 mask 不变；声明支持 mask 后
        # lstm/gru 的 mask_zero 才能把 padding 掩码透传到下游 RNN。
        self.supports_masking = True
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(max_len, embed_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        return inputs + self.position_embedding[tf.newaxis, :seq_len, :]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"max_len": self.max_len, "embed_dim": self.embed_dim})
        return config


def cfg_value(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def positive_vocab_size(vocab_sizes: Dict[str, int], name: str) -> int:
    size = int(vocab_sizes[name])
    if size <= 0:
        raise ValueError(f"vocab size for {name} must be positive")
    return size


def parse_units(value: Any, default: List[int]) -> List[int]:
    if value is None:
        return default
    if isinstance(value, str):
        units = [int(part) for part in value.replace(",", " ").split() if part]
        return units or default
    if isinstance(value, (list, tuple)):
        units = [int(part) for part in value]
        return units or default
    return default


def build_token_encoder(
    cfg: Any,
    vocab_sizes: Dict[str, int],
    *,
    mask_zero: bool = False,
) -> Tuple[Dict[str, Any], Any, Dict[str, int]]:
    """构建各字段嵌入并拼接投影。

    mask_zero 仅对能正确传播 mask 的循环模型（lstm/gru）启用；卷积类模型
    （cnn_lstm/tcn）和 transformer 的下游层对 mask 传播支持不一致，保持关闭。
    """
    max_seq_len = int(cfg_value(cfg, "max_seq_len", 64))
    model_dim = int(cfg_value(cfg, "model_dim", 128))

    sizes = {
        "module": positive_vocab_size(vocab_sizes, "module"),
        "function": positive_vocab_size(vocab_sizes, "function"),
        "offset": positive_vocab_size(vocab_sizes, "offset"),
        "node": positive_vocab_size(vocab_sizes, "node"),
        "ctrl_type": positive_vocab_size(vocab_sizes, "ctrl_type"),
        "icount": positive_vocab_size(vocab_sizes, "icount"),
    }
    inputs = {
        field: Input(shape=(max_seq_len,), dtype="int32", name=field)
        for field in INPUT_FIELDS
    }

    module_embedding = Embedding(
        input_dim=sizes["module"],
        output_dim=int(cfg_value(cfg, "module_emb_dim", 16)),
        mask_zero=mask_zero,
        name="module_embedding",
    )
    function_embedding = Embedding(
        input_dim=sizes["function"],
        output_dim=int(cfg_value(cfg, "function_emb_dim", 48)),
        mask_zero=mask_zero,
        name="function_embedding",
    )
    offset_embedding = Embedding(
        input_dim=sizes["offset"],
        output_dim=int(cfg_value(cfg, "offset_emb_dim", 32)),
        mask_zero=mask_zero,
        name="offset_embedding",
    )
    ctrl_type_embedding = Embedding(
        input_dim=sizes["ctrl_type"],
        output_dim=int(cfg_value(cfg, "ctrl_type_emb_dim", 8)),
        mask_zero=mask_zero,
        name="ctrl_type_embedding",
    )
    icount_embedding = Embedding(
        input_dim=sizes["icount"],
        output_dim=int(cfg_value(cfg, "icount_emb_dim", 8)),
        mask_zero=mask_zero,
        name="icount_embedding",
    )

    embedded = [
        module_embedding(inputs["src_module"]),
        function_embedding(inputs["src_ctrl_func"]),
        offset_embedding(inputs["src_ctrl_off"]),
        ctrl_type_embedding(inputs["ctrl_type"]),
        module_embedding(inputs["dst_module"]),
        function_embedding(inputs["dst_func"]),
        offset_embedding(inputs["dst_off"]),
        icount_embedding(inputs["icount"]),
    ]
    x = Concatenate(name="field_concat")(embedded)
    x = Dense(model_dim, activation="relu", name="field_projection")(x)
    x = LearnedPositionEmbedding(max_seq_len, model_dim, name="position_embedding")(x)
    return inputs, x, sizes


def prediction_heads(x: Any, sizes: Dict[str, int]) -> Dict[str, Any]:
    logits_dst_node = Dense(sizes["node"], dtype="float32", name="next_dst_node_logits")(x)
    logits_ctrl_type = Dense(sizes["ctrl_type"], dtype="float32", name="next_ctrl_type_logits")(x)
    return {
        "next_dst_node": Activation("softmax", dtype="float32", name="next_dst_node")(logits_dst_node),
        "next_ctrl_type": Activation("softmax", dtype="float32", name="next_ctrl_type")(logits_ctrl_type),
    }
