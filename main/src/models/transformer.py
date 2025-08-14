# src/models/transformer.py
"""
Keras Transformer for single-step time-series forecasting with delta prediction.
"""

from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="TimeSeries")
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        seq_len = min(self.max_len, int(input_shape[-2]))
        d_model = int(input_shape[-1])

        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32))
        )
        sin_vals = tf.sin(position * div_term)
        cos_vals = tf.cos(position * div_term)
        pe = tf.reshape(
            tf.stack([sin_vals, cos_vals], axis=-1), (seq_len, -1)
        )[:, :d_model]

        self.pe = tf.Variable(pe, trainable=False, name="pe")
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, D)
        t = tf.shape(x)[1]
        return x + self.pe[:t, :]

    def get_config(self):
        return {"max_len": self.max_len, **super().get_config()}


@register_keras_serializable(package="TimeSeries")
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = keras.Sequential(
            [layers.Dense(dff, activation="relu"), layers.Dense(d_model)]
        )
        self.dropout2 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False, mask: Optional[tf.Tensor] = None):
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training)
        out1 = self.norm1(x + self.dropout1(attn_output, training=training))
        ffn_out = self.ffn(out1, training=training)
        out2 = self.norm2(out1 + self.dropout2(ffn_out, training=training))
        return out2


@register_keras_serializable(package="TimeSeries", name="Transformer")
class Transformer(keras.Model):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dff: int,
        input_features: int,
        dropout_rate: float = 0.1,
        close_index: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_params = dict(
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dff=dff,
            input_features=input_features,
            dropout_rate=dropout_rate,
            close_index=close_index
        )
        self.close_index = close_index
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.input_features = input_features
        self.dropout_rate = dropout_rate

        self.input_proj = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(max_len=max(1024, seq_len))
        self.encoders = [TransformerEncoderBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)
        self.pool = layers.GlobalAveragePooling1D()

        # Head now predicts a delta (change) from the last known close price
        self.head = keras.Sequential(
            [layers.Dense(dff, activation="relu"), layers.Dropout(dropout_rate), layers.Dense(1, activation="linear")]
        )

    def call(self, x, training=False):
        # x: (B, T, F)
        last_close = x[:, -1, self.close_index:self.close_index+1]  
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        for blk in self.encoders:
            z = blk(z, training=training)
        z = self.dropout(z, training=training)
        z = self.pool(z)
        delta = self.head(z, training=training)
        return last_close + delta  # predict next close as last close + change

    def get_config(self):
        return {**self._init_params, **super().get_config()}

    @classmethod
    def from_config(cls, config):
        known = dict(config)
        known.pop("compile_config", None)
        return cls(**known)

    def get_init_params(self):
        return dict(self._init_params)
