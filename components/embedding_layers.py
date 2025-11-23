"""Embedding and regularization layers for OmniTFT.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import tensorflow as tf

# Layer definitions.
concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply


class ShockAwareAlignmentLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_size, num_encoder_steps, lambda_shock=1e-1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_encoder_steps = num_encoder_steps
        self.lambda_shock = lambda_shock

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):

        temporal_features, attention_weights = inputs

        if training:
            loss_shock = self._compute_shock_loss(temporal_features, attention_weights)
            self.add_loss(self.lambda_shock * loss_shock)

        return temporal_features

    def _compute_shock_loss(self, temporal_features, attn):
        T = tf.shape(temporal_features)[1]
        enc = self.num_encoder_steps
        dec_mask = tf.concat([tf.zeros([enc], tf.float32),
                              tf.ones([T - enc], tf.float32)], axis=0)
        dec_mask_bt = tf.reshape(dec_mask, [1, T])

        def _diff_l2_per_timestep(x, lag):
            x_f = x[:, lag:, :] - x[:, :-lag, :]
            pad = tf.zeros_like(x[:, :lag, :])
            diff = tf.concat([pad, x_f], axis=1)
            return tf.norm(diff, ord='euclidean', axis=-1)

        shock_1 = _diff_l2_per_timestep(temporal_features, 1)
        shock_1_n = self._zscore(shock_1, axis=1)

        att_bt = tf.reduce_mean(attn, axis=0) if len(attn.shape) == 4 else attn
        ones_tt = tf.ones([T, T], tf.float32)
        W = 3
        bandW = tf.linalg.band_part(ones_tt, W, 0)
        diagI = tf.linalg.band_part(ones_tt, 0, 0)
        win_mask = bandW - diagI
        att_local = tf.reduce_sum(att_bt * win_mask, axis=-1)
        att_local_n = self._zscore(att_local, axis=1)
        mse_dec = tf.reduce_mean(tf.square((att_local_n - shock_1_n)) * dec_mask_bt)
        return mse_dec

    def _zscore(self, x, axis=-1, eps=1e-6):
        m = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.math.reduce_std(x, axis=axis, keepdims=True) + eps
        return (x - m) / s


class GroupSparsityLayer(tf.keras.layers.Layer):
    """Imposing group sparsity constraints on variable selection weights"""

    def __init__(self, lambda_group=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.lambda_group = lambda_group

    def call(self, inputs, training=None):
        if len(inputs) == 6:
            features, hist_flags, fut_flags, unknown_inputs, known_combined_layer, obs_inputs = inputs
        else:
            # Fallback for compatibility
            features = inputs[0]
            hist_flags = inputs[1] if len(inputs) > 1 else None
            fut_flags = inputs[2] if len(inputs) > 2 else None
            unknown_inputs = None
            known_combined_layer = None
            obs_inputs = None

        if training and hist_flags is not None:
            loss_group = self._compute_group_loss(hist_flags, fut_flags,
                                                 unknown_inputs, known_combined_layer, obs_inputs)
            self.add_loss(self.lambda_group * loss_group)

        return features

    def _compute_group_loss(self, hist_flags, fut_flags, unknown_inputs, known_combined_layer, obs_inputs):
        hist_flags_3d = tf.squeeze(hist_flags, axis=2)
        fut_flags_3d = tf.squeeze(fut_flags, axis=2)

        _EPS = 1e-9

        unknown_n = tf.shape(unknown_inputs)[-1] if unknown_inputs is not None else tf.constant(0, tf.int32)
        known_n = tf.shape(known_combined_layer)[-1] if known_combined_layer is not None else tf.constant(0, tf.int32)
        obs_n = tf.shape(obs_inputs)[-1] if obs_inputs is not None else tf.constant(0, tf.int32)
        total_n = unknown_n + known_n + obs_n

        idx = tf.range(total_n)
        mask_unknown = tf.cast(idx < unknown_n, tf.float32)
        mask_known = tf.cast((idx >= unknown_n) & (idx < unknown_n + known_n), tf.float32)
        mask_obs = tf.cast(idx >= unknown_n + known_n, tf.float32)
        group_mask_hist = tf.stack([mask_unknown, mask_known, mask_obs], axis=0)
        hist_groups = tf.einsum('btn,gn->btg', hist_flags_3d, group_mask_hist)

        future_known_sum = tf.reduce_sum(fut_flags_3d, axis=-1, keepdims=True)
        future_groups = tf.concat([
            tf.zeros_like(future_known_sum),
            future_known_sum,
            tf.zeros_like(future_known_sum)
        ], axis=-1)

        def _entropy_3(p_bt3):
            s = tf.reduce_sum(p_bt3, axis=-1, keepdims=True)
            p = tf.clip_by_value(p_bt3 / (s + _EPS), _EPS, 1.0)
            ent = -tf.reduce_sum(p * tf.math.log(p), axis=-1)
            return tf.reduce_mean(ent)

        return 0.5 * (_entropy_3(hist_groups) + _entropy_3(future_groups))


class FrequencyAwareEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, embedding_layers, category_counts, input_size, lambda_embed=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layers = embedding_layers
        self.category_counts = category_counts
        self.input_size = input_size
        self.lambda_embed = lambda_embed

    def call(self, inputs, training=None):

        all_inputs = inputs[0]

        if training and len(self.category_counts) > 0:
            loss_embed = self._compute_embed_loss(all_inputs)
            self.add_loss(self.lambda_embed * loss_embed)

        return all_inputs

    def _compute_embed_loss(self, all_inputs):
        loss_embed = 0.0
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        cat_raw = all_inputs[:, :, num_regular_variables:]
        cat_raw = tf.cast(cat_raw, tf.int32)

        for i, vocab_size in enumerate(self.category_counts):
            ids = tf.reshape(cat_raw[:, :, i], [-1])
            hist = tf.math.bincount(ids, minlength=vocab_size,
                                   maxlength=vocab_size, dtype=tf.float32)
            p = hist / (tf.reduce_sum(hist) + 1e-9)
            inv_sqrt = tf.math.rsqrt(p + 1e-9)

            emb_layer = self.embedding_layers[i]
            if hasattr(emb_layer, 'layers') and len(emb_layer.layers) > 0:
                # Sequential with layers
                _cands = [l for l in emb_layer.layers if isinstance(l, tf.keras.layers.Embedding)]
                if len(_cands) > 0:
                    W = _cands[0].embeddings
                else:
                    W = emb_layer.layers[-1].embeddings if hasattr(emb_layer.layers[-1], 'embeddings') else None
            else:
                # Direct embedding
                W = emb_layer.embeddings if hasattr(emb_layer, 'embeddings') else None

            if W is not None:
                w_norm2 = tf.reduce_sum(tf.square(W), axis=1)
                reg_i = tf.reduce_mean(inv_sqrt * w_norm2)
                loss_embed += reg_i

        return loss_embed


def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
    linear = tf.keras.layers.Dense(
        size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear


def apply_mlp(inputs,
              hidden_size,
              output_size,
              output_activation=None,
              hidden_activation='tanh',
              use_time_distributed=False):
    """Apply multi-layer perceptron."""
    if use_time_distributed:
        hidden = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_size, activation=hidden_activation))(
                inputs)
        return tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(output_size, activation=output_activation))(
                hidden)
    else:
        hidden = tf.keras.layers.Dense(
            hidden_size, activation=hidden_activation)(
                inputs)
        return tf.keras.layers.Dense(
            output_size, activation=output_activation)(
                hidden)


def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):

    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation=activation))(
                x)
        gated_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(
                x)
    else:
        activation_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation=activation)(
                x)
        gated_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation='sigmoid')(
                x)

    return tf.keras.layers.Multiply()([activation_layer,
                                       gated_layer]), gated_layer


def add_and_norm(x_list):
    """Add and normalize layers."""
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp


def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
    """Gated Residual Network."""
    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    # Apply feedforward network
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
            x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False)(
                additional_context)
    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
            hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None)

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])


class TFTDataCache(object):

    _data_cache = {}

    @classmethod
    def update(cls, data, key):

        cls._data_cache[key] = data

    @classmethod
    def get(cls, key):
        """Returns data stored at key location."""
        return cls._data_cache[key].copy()

    @classmethod
    def contains(cls, key):
        """Retuns boolean indicating whether key is present in cache."""

        return key in cls._data_cache
