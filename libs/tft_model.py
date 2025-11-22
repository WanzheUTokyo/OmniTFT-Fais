"""OmniTFT.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import json
import os
import shutil
from tensorflow.keras.metrics import MeanAbsoluteError
import data_formatters.base
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow as tf

# Layer definitions.
concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda

# Default input types.
InputTypes = data_formatters.base.InputTypes


class ShockAwareAlignmentLayer(tf.keras.layers.Layer):
    """Transparent Layer: Calculate and add shock alignment loss"""
    
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
    """Frequency-aware regularisation for embedded additions"""
    
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
    """Returns simple Keras linear layer.

    Args:
      size: Output size
      activation: Activation function to apply if required
      use_time_distributed: Whether to apply layer across time
      use_bias: Whether bias should be included in layer
    """
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
    """Applies a Gated Linear Unit (GLU) to an input.

    Args:
      x: Input to gating layer
      hidden_layer_size: Dimension of GLU
      dropout_rate: Dropout rate to apply if any
      use_time_distributed: Whether to apply across time
      activation: Activation function to apply to the linear feature transform if
        necessary

    Returns:
      Tuple of tensors for: (GLU output, gate)
    """

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


# Attention Components.
def get_decoder_mask(self_attn_inputs):

    len_s = tf.shape(input=self_attn_inputs)[1]
    bs = tf.shape(input=self_attn_inputs)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class ScaledDotProductAttention():


    def __init__(self, attn_dropout=0.0):
        self.dropout = Dropout(attn_dropout)
        self.activation = Activation('softmax')

    def __call__(self, q, k, v, mask):
        """Applies scaled dot product attention.

        Args:
          q: Queries
          k: Keys
          v: Values
          mask: Masking if required -- sets softmax to very large value

        Returns:
          Tuple of (layer outputs, attention weights)
        """
        temper = tf.sqrt(tf.cast(tf.shape(input=k)[-1], dtype='float32'))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)(
            [q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(
                mask)  # setting to infinity
            attn = Add()([attn, mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class InterpretableMultiHeadAttention():


    def __init__(self, n_head, d_model, dropout):
        """Initialises layer.

        Args:
          n_head: Number of heads
          d_model: TFT state dimensionality
          dropout: Dropout discard rate
        """

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        # Use same value layer to facilitate interp
        vs_layer = Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(Dense(d_k, use_bias=False))
            self.ks_layers.append(Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.attention = ScaledDotProductAttention()
        self.w_o = Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """Applies interpretable multihead attention.

        Using T to denote the number of time steps fed into the transformer.

        Args:
          q: Query tensor of shape=(?, T, d_model)
          k: Key of shape=(?, T, d_model)
          v: Values of shape=(?, T, d_model)
          mask: Masking if required with shape=(?, T, T)

        Returns:
          Tuple of (layer outputs, attention weights)
        """
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = K.stack(heads) if n_head > 1 else heads[0]
        attn = K.stack(attns)

        outputs = K.mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = Dropout(self.dropout)(outputs)  # output dropout

        return outputs, attn


class TFTDataCache(object):

    _data_cache = {}

    @classmethod
    def update(cls, data, key):
        """Updates cached data.

        Args:
          data: Source to update
          key: Key to dictionary location
        """
        cls._data_cache[key] = data

    @classmethod
    def get(cls, key):
        """Returns data stored at key location."""
        return cls._data_cache[key].copy()

    @classmethod
    def contains(cls, key):
        """Retuns boolean indicating whether key is present in cache."""

        return key in cls._data_cache


class TemporalFusionTransformer(object):

    def __init__(self, raw_params, use_cudnn=False):
        """Builds TFT from parameters.

        Args:
          raw_params: Parameters to define TFT
          use_cudnn: Whether to use CUDNN GPU optimised LSTM
        """

        self.name = self.__class__.__name__

        params = dict(raw_params)  # copy locally

        # Data parameters
        self.time_steps = int(params['total_time_steps'])
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        self.category_counts = json.loads(str(params['category_counts']))
        self.n_multiprocessing_workers = int(params['multiprocessing_workers'])

        # Relevant indices for TFT
        self._input_obs_loc = json.loads(str(params['input_obs_loc']))
        self._static_input_loc = json.loads(str(params['static_input_loc']))
        self._known_regular_input_idx = json.loads(
            str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(
            str(params['known_categorical_inputs']))

        self.column_definition = params['column_definition']

        # Network params
        self.quantiles = [0.1, 0.5, 0.9]
        self.use_cudnn = use_cudnn  # Whether to use GPU optimised LSTM
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self.learning_rate = float(params['learning_rate'])
        self.minibatch_size = int(params['minibatch_size'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.num_stacks = int(params['stack_size'])
        self.num_heads = int(params['num_heads'])
        self.lambda_shock = float(params.get('lambda_shock', 1e-3))
        self.lambda_group = float(params.get('lambda_group', 1e-4))
        self.lambda_embed = float(params.get('lambda_embed', 1e-6))
        
        # Serialisation options
        self._temp_folder = os.path.join(params['model_folder'], 'tmp')
        self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        print('*** {} params ***'.format(self.name))
        for k in params:
            print('# {} = {}'.format(k, params[k]))
        
        # Store loss history
        self.train_loss_history = []
        self.val_loss_history = []
        self.history = None
        
        # Build model
        self.model = self.build_model()

    def get_tft_embeddings(self, all_inputs):
        """Transforms raw inputs to embeddings.

        Applies linear transformation onto continuous variables and uses embeddings
        for categorical variables.

        Args:
          all_inputs: Inputs to transform

        Returns:
          Tensors for transformed inputs.
        """

        time_steps = self.time_steps

        # Sanity checks
        for i in self._known_regular_input_idx:
            if i in self._input_obs_loc:
                raise ValueError('Observation cannot be known a priori!')
        for i in self._input_obs_loc:
            if i in self._static_input_loc:
                raise ValueError('Observation cannot be static!')

        if all_inputs.get_shape().as_list()[-1] != self.input_size:
            raise ValueError(
                'Illegal number of inputs! Inputs observed={}, expected={}'.format(
                    all_inputs.get_shape().as_list()[-1], self.input_size))

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
            self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        embeddings = []
        for i in range(num_categorical_variables):

            embedding = tf.keras.Sequential([
                tf.keras.layers.InputLayer([time_steps]),
                tf.keras.layers.Embedding(
                    self.category_counts[i],
                    embedding_sizes[i],
                    input_length=time_steps,
                    dtype=tf.float32)
            ])
            embeddings.append(embedding)
        
        # Keep embedding layer objects for aux embedding regularizer
        self._cat_embedding_layers = embeddings

        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :num_regular_variables], \
            all_inputs[:, :, num_regular_variables:]

        embedded_inputs = [
            embeddings[i](categorical_inputs[Ellipsis, i])
            for i in range(num_categorical_variables)
        ]

        # Static inputs
        if self._static_input_loc:
            static_inputs = [tf.keras.layers.Dense(self.hidden_layer_size)(
                regular_inputs[:, 0, i:i + 1]) for i in range(num_regular_variables)
                if i in self._static_input_loc] \
                + [embedded_inputs[i][:, 0, :]
                   for i in range(num_categorical_variables)
                   if i + num_regular_variables in self._static_input_loc]
            static_inputs = tf.keras.backend.stack(static_inputs, axis=1)

        else:
            static_inputs = None

        def convert_real_to_embedding(x):
            """Applies linear transformation for time-varying inputs."""
            return tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.hidden_layer_size))(
                    x)

        # Targets
        obs_inputs = tf.keras.backend.stack([
            convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1])
            for i in self._input_obs_loc
        ],
            axis=-1)

        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx \
                    and i + num_regular_variables not in self._input_obs_loc:
                e = embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx \
                    and i not in self._input_obs_loc:
                e = convert_real_to_embedding(
                    regular_inputs[Ellipsis, i:i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = tf.keras.backend.stack(
                unknown_inputs + wired_embeddings, axis=-1)
        else:
            unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = [
            convert_real_to_embedding(regular_inputs[Ellipsis, i:i + 1])
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        known_combined_layer = tf.keras.backend.stack(
            known_regular_inputs + known_categorical_inputs, axis=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""

        return utils.get_single_col_by_input_type(input_type,
                                                  self.column_definition)

    def training_data_cached(self):
        """Returns boolean indicating if training data has been cached."""

        return TFTDataCache.contains('train') and TFTDataCache.contains('valid')

    def cache_batched_data(self, data, cache_key, num_samples=-1):
        """Batches and caches data once for using during training.

        Args:
          data: Data to batch and cache
          cache_key: Key used for cache
          num_samples: Maximum number of samples to extract (-1 to use all data)
        """

        if num_samples > 0:
            TFTDataCache.update(
                self._batch_sampled_data(data, max_samples=num_samples), cache_key)
        else:
            TFTDataCache.update(self._batch_data(data), cache_key)

    def _batch_sampled_data(self, data, max_samples):
        """Samples segments into a compatible format.

        Args:
          data: Sources data to sample and batch
          max_samples: Maximum number of samples in batch

        Returns:
          Dictionary of batched data with the maximum samples specified.
        """

        if max_samples < 1:
            raise ValueError(
                'Illegal number of samples specified! samples={}'.format(max_samples))

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)

        data.sort_values(by=[id_col, time_col], inplace=True)

        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [
                    (identifier, self.time_steps + i)
                    for i in range(num_entries - self.time_steps + 1)
                ]
            split_data_map[identifier] = df

        inputs = np.zeros((max_samples, self.time_steps, self.input_size))
        outputs = np.zeros((max_samples, self.time_steps, self.output_size))
        time = np.empty((max_samples, self.time_steps, 1), dtype=object)
        identifiers = np.empty((max_samples, self.time_steps, 1), dtype=object)

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            ranges = [
                valid_sampling_locations[i] for i in np.random.choice(
                    len(valid_sampling_locations), max_samples, replace=False)
            ]
        else:
            ranges = valid_sampling_locations

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        for i, tup in enumerate(ranges):
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx -
                                                     self.time_steps:start_idx]
            inputs[i, :, :] = sliced[input_cols]
            outputs[i, :, :] = sliced[[target_col]]
            time[i, :, 0] = sliced[time_col]
            identifiers[i, :, 0] = sliced[id_col]

        sampled_data = {
            'inputs': inputs,
            'outputs': outputs[:, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(outputs[:, self.num_encoder_steps:, :]),
            'time': time,
            'identifier': identifiers
        }

        return sampled_data

    def _batch_data(self, data):

        # Functions.
        def _batch_single_entity(input_data):
            time_steps = len(input_data)
            lags = self.time_steps
            x = input_data.values
            if time_steps >= lags:
                return np.stack(
                    [x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)

            else:
                return None

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        data_map = {}
        for _, sliced in data.groupby(id_col):

            col_mappings = {
                'identifier': [id_col],
                'time': [time_col],
                'outputs': [target_col],
                'inputs': input_cols
            }

            for k in col_mappings:
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())

                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)

        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

        # Shorten target so we only get decoder steps
        data_map['outputs'] = data_map['outputs'][:,
                                                  self.num_encoder_steps:, :]

        active_entries = np.ones_like(data_map['outputs'])
        if 'active_entries' not in data_map:
            data_map['active_entries'] = active_entries
        else:
            data_map['active_entries'].append(active_entries)

        return data_map

    def _get_active_locations(self, x):
        """Formats sample weights for Keras training."""
        return (np.sum(x, axis=-1) > 0.0) * 1.0

    def _build_base_graph(self):
        """Returns graph defining layers of the TFT."""

        # Size definitions.
        time_steps = self.time_steps
        combined_input_size = self.input_size
        encoder_steps = self.num_encoder_steps

        # Inputs.
        all_inputs = tf.keras.layers.Input(
            shape=(
                time_steps,
                combined_input_size,
            ))
        
        masked_inputs = tf.keras.layers.Masking(mask_value=-100)(all_inputs)
        
        # Add frequency embedding regularization layer (early application)
        freq_layer = FrequencyAwareEmbeddingLayer(
            self._cat_embedding_layers if hasattr(self, '_cat_embedding_layers') else [],
            self.category_counts,
            self.input_size,
            self.lambda_embed
        )
        masked_inputs = freq_layer([masked_inputs, None])

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs \
            = self.get_tft_embeddings(masked_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = concat([
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ],
                axis=-1)
        else:
            historical_inputs = concat([
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ],
                axis=-1)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        def static_combine_and_mask(embedding):

            # Add temporal features
            _, num_static, _ = embedding.get_shape().as_list()

            flatten = tf.keras.layers.Flatten()(embedding)

            # Nonlinear transformation with gated residual network.
            mlp_outputs = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_static,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=None)

            sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
            sparse_weights = K.expand_dims(sparse_weights, axis=-1)

            trans_emb_list = []
            for i in range(num_static):
                e = gated_residual_network(
                    embedding[:, i:i + 1, :],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False)
                trans_emb_list.append(e)

            transformed_embedding = concat(trans_emb_list, axis=1)

            combined = tf.keras.layers.Multiply()(
                [sparse_weights, transformed_embedding])

            static_vec = K.sum(combined, axis=1)

            return static_vec, sparse_weights

        static_encoder, static_weights = static_combine_and_mask(static_inputs)

        static_context_variable_selection = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_enrichment = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_h = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_c = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)

        def lstm_combine_and_mask(embedding):
            """Apply temporal variable selection networks.

            Args:
              embedding: Transformed inputs.

            Returns:
              Processed tensor outputs.
            """

            # Add temporal features
            _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()

            flatten = K.reshape(embedding,
                                [-1, time_steps, embedding_dim * num_inputs])

            expanded_static_context = K.expand_dims(
                static_context_variable_selection, axis=1)

            # Variable selection weights
            mlp_outputs, static_gate = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_inputs,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                additional_context=expanded_static_context,
                return_gate=True)

            sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
            sparse_weights = tf.expand_dims(sparse_weights, axis=2)

            # Non-linear Processing & weight application
            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(
                    embedding[Ellipsis, i],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True)
                trans_emb_list.append(grn_output)

            transformed_embedding = stack(trans_emb_list, axis=-1)

            combined = tf.keras.layers.Multiply()(
                [sparse_weights, transformed_embedding])
            temporal_ctx = K.sum(combined, axis=-1)

            return temporal_ctx, sparse_weights, static_gate

        historical_features, historical_flags, _ = lstm_combine_and_mask(
            historical_inputs)
        future_features, future_flags, _ = lstm_combine_and_mask(future_inputs)

        def get_lstm(return_state):
            """Returns LSTM cell initialized with default parameters."""
            lstm = tf.keras.layers.LSTM(
                self.hidden_layer_size,
                return_sequences=True,
                return_state=return_state,
                stateful=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True
            )
            return lstm
        
        history_lstm, state_h, state_c \
            = get_lstm(return_state=True)(historical_features,
                                          initial_state=[static_context_state_h,
                                                         static_context_state_c])

        future_lstm = get_lstm(return_state=False)(
            future_features, initial_state=[state_h, state_c])

        lstm_layer = concat([history_lstm, future_lstm], axis=1)

        # Apply gated skip connection
        input_embeddings = concat(
            [historical_features, future_features], axis=1)

        lstm_layer, _ = apply_gating_layer(
            lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None)
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])
        
        # Add Group Sparsity Layer
        group_layer = GroupSparsityLayer(lambda_group=self.lambda_group)
        temporal_feature_layer = group_layer([
            temporal_feature_layer, 
            historical_flags, 
            future_flags,
            unknown_inputs,
            known_combined_layer,
            obs_inputs
        ])

        # Static enrichment layers
        expanded_static_context = K.expand_dims(
            static_context_enrichment, axis=1)
        enriched, _ = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context,
            return_gate=True)

        # Decoder self attention
        self_attn_layer = InterpretableMultiHeadAttention(
            self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate)

        mask = get_decoder_mask(enriched)
        x, self_att \
            = self_attn_layer(enriched, enriched, enriched,
                              mask=mask)
        
        # Add Shock-Aware Alignment Layer
        shock_layer = ShockAwareAlignmentLayer(
            self.hidden_layer_size,
            self.num_encoder_steps,
            self.lambda_shock
        )
        x = shock_layer([x, self_att])

        x, _ = apply_gating_layer(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            activation=None)
        x = add_and_norm([x, enriched])

        # Nonlinear processing on outputs
        decoder = gated_residual_network(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True)

        # Final skip connection
        decoder, _ = apply_gating_layer(
            decoder, self.hidden_layer_size, activation=None)
        transformer_layer = add_and_norm([decoder, temporal_feature_layer])

        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }

        return transformer_layer, all_inputs, attention_components

    def build_model(self):
        """Build model and defines training losses.

        Returns:
          Fully defined Keras model.
        """

        with tf.compat.v1.variable_scope(self.name):

            transformer_layer, all_inputs, attention_components \
                = self._build_base_graph()

            outputs = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.output_size * len(self.quantiles)))(
                    transformer_layer[Ellipsis, self.num_encoder_steps:, :])

            self._attention_components = attention_components

            adam = tf.compat.v1.keras.optimizers.Adam(
                learning_rate=self.learning_rate, clipnorm=self.max_gradient_norm)

            model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

            valid_quantiles = self.quantiles
            output_size = self.output_size

            class QuantileLossCalculator(object):

                def __init__(self, quantiles):

                    self.quantiles = quantiles

                def quantile_loss(self, a, b):

                    quantiles_used = set(self.quantiles)

                    loss = 0.
                    for i, quantile in enumerate(valid_quantiles):
                        if quantile in quantiles_used:
                            loss += utils.tensorflow_quantile_loss(
                                a[Ellipsis, output_size *
                                    i:output_size * (i + 1)],
                                b[Ellipsis, output_size * i:output_size * (i + 1)], quantile)

                    return loss

            quantile_loss = QuantileLossCalculator(
                valid_quantiles).quantile_loss

            model.compile(
                loss=quantile_loss, optimizer=adam, sample_weight_mode='temporal', 
                metrics=[MeanAbsoluteError(name='mae')])

            self._input_placeholder = all_inputs

        return model

    def fit(self, train_df=None, valid_df=None):


        print('*** Fitting {} ***'.format(self.name))

        # Add relevant callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=1e-4),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_keras_saved_path(self._temp_folder),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        print('Getting batched_data')
        if train_df is None:
            train_data = TFTDataCache.get('train')
        else:
            train_data = self._batch_data(train_df)

        if valid_df is None:
            valid_data = TFTDataCache.get('valid')
        else:
            valid_data = self._batch_data(valid_df)

        print('Using keras standard fit')

        def _unpack(data):
            return data['inputs'], data['outputs'], \
                self._get_active_locations(data['active_entries'])

        # Unpack without sample weights
        data, labels, active_flags = _unpack(train_data)
        val_data, val_labels, val_flags = _unpack(valid_data)

        all_callbacks = callbacks
        
        self.history = self.model.fit(
            x=data,
            y=np.concatenate([labels, labels, labels], axis=-1),
            sample_weight=active_flags,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            validation_data=(val_data,
                             np.concatenate([val_labels, val_labels, val_labels],
                                            axis=-1), val_flags),
            callbacks=all_callbacks,
            shuffle=True,
            use_multiprocessing=True,
            workers=self.n_multiprocessing_workers,
            verbose=2)

        # Load best checkpoint again
        tmp_checkpoint = self.get_keras_saved_path(self._temp_folder)
        if os.path.exists(tmp_checkpoint):
            self.load(self._temp_folder, use_keras_loadings=True)
        else:
            print('Cannot load from {}, skipping ...'.format(self._temp_folder))

        return self.history

    def evaluate(self, data=None, eval_metric='loss'):


        if data is None:
            print('Using cached validation data')
            raw_data = TFTDataCache.get('valid')
        else:
            raw_data = self._batch_data(data)

        inputs = raw_data['inputs']
        outputs = raw_data['outputs']
        active_entries = self._get_active_locations(raw_data['active_entries'])

        metric_values = self.model.evaluate(
            x=inputs,
            y=np.concatenate([outputs, outputs, outputs], axis=-1),
            sample_weight=active_entries,
            workers=16,
            use_multiprocessing=True)

        metrics = pd.Series(metric_values, self.model.metrics_names)

        return metrics[eval_metric]

    def predict(self, df, return_targets=False):

        data = self._batch_data(df)

        inputs = data['inputs']
        time = data['time']
        identifier = data['identifier']
        outputs = data['outputs']

        combined = self.model.predict(
            inputs,
            workers=16,
            use_multiprocessing=True,
            batch_size=self.minibatch_size)

        # Format output_csv
        if self.output_size != 1:
            raise NotImplementedError(
                'Current version only supports 1D targets!')

        def format_outputs(prediction):
            """Returns formatted dataframes for prediction."""

            flat_prediction = pd.DataFrame(
                prediction[:, :, 0],
                columns=[
                    't+{}'.format(i)
                    for i in range(self.time_steps - self.num_encoder_steps)
                ])
            cols = list(flat_prediction.columns)
            flat_prediction['forecast_time'] = time[:,
                                                    self.num_encoder_steps - 1, 0]
            flat_prediction['identifier'] = identifier[:, 0, 0]

            # Arrange in order
            return flat_prediction[['forecast_time', 'identifier'] + cols]

        # Extract predictions for each quantile into different entries
        process_map = {
            'p{}'.format(int(q * 100)):
            combined[Ellipsis, i * self.output_size:(i + 1) * self.output_size]
            for i, q in enumerate(self.quantiles)
        }

        if return_targets:
            # Add targets if relevant
            process_map['targets'] = outputs

        return {k: format_outputs(process_map[k]) for k in process_map}

    def get_attention(self, df):

        data = self._batch_data(df)
        inputs = data['inputs']
        identifiers = data['identifier']
        time = data['time']

        def get_batch_attention_weights(input_batch):
            """Returns weights for a given minibatch of data."""
            input_placeholder = self._input_placeholder
            attention_weights = {}
            for k in self._attention_components:
                attention_weight = tf.compat.v1.keras.backend.get_session().run(
                    self._attention_components[k],
                    {input_placeholder: input_batch.astype(np.float32)})
                attention_weights[k] = attention_weight
            return attention_weights

        # Compute number of batches
        batch_size = self.minibatch_size
        n = inputs.shape[0]
        num_batches = n // batch_size
        if n - (num_batches * batch_size) > 0:
            num_batches += 1

        # Split up inputs into batches
        batched_inputs = [
            inputs[i * batch_size:(i + 1) * batch_size, Ellipsis]
            for i in range(num_batches)
        ]

        # Get attention weights, while avoiding large memory increases
        attention_by_batch = [
            get_batch_attention_weights(batch) for batch in batched_inputs
        ]
        attention_weights = {}
        for k in self._attention_components:
            attention_weights[k] = []
            for batch_weights in attention_by_batch:
                attention_weights[k].append(batch_weights[k])

            if len(attention_weights[k][0].shape) == 4:
                tmp = np.concatenate(attention_weights[k], axis=1)
            else:
                tmp = np.concatenate(attention_weights[k], axis=0)

            del attention_weights[k]
            gc.collect()
            attention_weights[k] = tmp

        attention_weights['identifiers'] = identifiers[:, 0, 0]
        attention_weights['time'] = time[:, :, 0]

        return attention_weights

    # Serialisation.
    def reset_temp_folder(self):
        """Deletes and recreates folder with temporary Keras training outputs."""
        print('Resetting temp folder...')
        utils.create_folder_if_not_exist(self._temp_folder)
        shutil.rmtree(self._temp_folder)
        os.makedirs(self._temp_folder)

    def get_keras_saved_path(self, model_folder):
        """Returns path to keras checkpoint."""
        return os.path.join(model_folder, '{}.check'.format(self.name))

    def save_config(self, config_path):
        """Save model configuration to a JSON file."""
        config = {
            'name': self.name,
            'time_steps': self.time_steps,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'category_counts': self.category_counts,
            'n_multiprocessing_workers': self.n_multiprocessing_workers,
            '_input_obs_loc': self._input_obs_loc,
            '_static_input_loc': self._static_input_loc,
            '_known_regular_input_idx': self._known_regular_input_idx,
            '_known_categorical_input_idx': self._known_categorical_input_idx,
            'column_definition': self.column_definition,
            'quantiles': self.quantiles,
            'use_cudnn': self.use_cudnn,
            'hidden_layer_size': self.hidden_layer_size,
            'dropout_rate': self.dropout_rate,
            'max_gradient_norm': self.max_gradient_norm,
            'learning_rate': self.learning_rate,
            'minibatch_size': self.minibatch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'num_encoder_steps': self.num_encoder_steps,
            'num_stacks': self.num_stacks,
            'num_heads': self.num_heads,
            'lambda_shock': self.lambda_shock,
            'lambda_group': self.lambda_group,
            'lambda_embed': self.lambda_embed
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Model configuration saved to {config_path}")

    def save(self, model_folder):
        """Saves optimal TFT weights and configuration."""
        # Save model weights
        utils.save(
            tf.compat.v1.keras.backend.get_session(),
            model_folder,
            cp_name=self.name,
            scope=self.name)
        # Save model configuration
        config_path = os.path.join(model_folder, 'model_config.json')
        self.save_config(config_path)

    def load(self, model_folder, use_keras_loadings=False):
        """Loads TFT weights.

        Args:
          model_folder: Folder containing serialized models.
          use_keras_loadings: Whether to load from Keras checkpoint.

        Returns:

        """
        if use_keras_loadings:
            # Loads temporary Keras model saved during training.
            serialisation_path = self.get_keras_saved_path(model_folder)
            print('Loading model from {}'.format(serialisation_path))
            self.model.load_weights(serialisation_path)
        else:
            # Loads tensorflow graph for optimal models.
            utils.load(
                tf.compat.v1.keras.backend.get_session(),
                model_folder,
                cp_name=self.name,
                scope=self.name)

    @classmethod
    def get_hyperparm_choices(cls):
        """Returns hyperparameter ranges for random search."""
        return {
            'dropout_rate': [0.2, 0.3, 0.4, 0.5], 
            'hidden_layer_size': [64, 128, 256],
            'minibatch_size': [128, 256, 512],
            'learning_rate': [0.0008, 0.0002, 0.00002, 0.000005], 
            'max_gradient_norm': [0.01],
            'num_heads': [4, 6],
            'stack_size': [3],
            'lambda_shock': [1e-3, 1e-2, 1e-1],
            'lambda_group': [1e-4, 1e-3, 1e-2],
            'lambda_embed': [1e-5, 1e-4, 1e-3]
        }