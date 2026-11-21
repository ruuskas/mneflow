#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:33:35 2024

@author: eero.saarro@aalto.fi
"""

import os
from collections import defaultdict
from copy import deepcopy
from typing import cast

import numpy as np
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf
import keras
from keras.layers import (Dense, Flatten, Dropout, BatchNormalization,
                                     Conv2D, SpatialDropout2D, Conv3D,
                                     SpatialDropout3D, Lambda, DepthwiseConv2D)
from keras import initializers

# working_directory = Path("/m/nbe/scratch/restmeg/eero/code/mneflow/")
# os.chdir(working_directory)
#import mneflow
from .layers import SquareSum3d, WeightSum3d
from .layers import se_block
from .models import BaseModel
from .data import Dataset


@keras.saving.register_keras_serializable(package="mneflow")
class GaussianNoise(keras.layers.GaussianNoise):
    """keras.layers.GaussianNoise without its upper limit of 1 on stddev.

    The input noise of the FC models is in units of the inputs, which can
    be z-scored (SD 1), so SDs above 1 are useful. Otherwise the same
    layer (and layer name): additive N(0, stddev) noise, in training only.
    """

    def __init__(self, stddev, seed=None, **kwargs):
        if stddev < 0:
            raise ValueError(f"stddev must be >= 0, got {stddev}")
        # The parent checks 0 <= stddev <= 1 and creates the seed
        # generator if stddev > 0; then the actual stddev is set.
        super().__init__(stddev=min(stddev, 1.), seed=seed, **kwargs)
        self.stddev = stddev


def input_covariance(X):
    """Spatial covariance of connectivity inputs, per frequency.

    SquareSum3d first projects the rows of each input matrix,
    ``s[h, j] = sum_i X[h, i, j] w[i]``: the rows i are the channels and
    each (sample h, column j) pair is an observation. Its activation
    pattern (Haufe et al., 2014) is ``C @ w`` with the covariance of the
    rows over these observations,
    ``C[i, l] = sum_{h, j} X[h, i, j] X[h, l, j] / (n_samples - 1)``.
    X is centred over the samples only.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_nodes, n_nodes, n_freqs)

    Returns
    -------
    dcov : ndarray, shape (n_nodes, n_nodes, n_freqs)
        Symmetric and positive semidefinite for each frequency.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    return np.einsum('hijk, hljk -> ilk', X, X) / (X.shape[0] - 1)


def _fold_stability(data, threshold):
    """Consistency of spatial patterns across CV folds.

    Parameters
    ----------
    data : ndarray, shape (n_sources, n_folds)
        Standardized pattern of each fold.
    threshold : float or ndarray, shape (n_folds,)
        Sources with ``abs(data) > threshold`` count as top sources.

    Returns
    -------
    dict
        Mean over all pairs of folds of ``'jaccard'`` (Jaccard index of the
        top-source sets) and ``'abs_r'`` (absolute correlation of the
        patterns).
    """
    masks = np.abs(data) > threshold
    jaccard, abs_r = [], []
    n_folds = data.shape[-1]
    for i in range(n_folds - 1):
        for j in range(i + 1, n_folds):
            union = np.sum(masks[:, i] | masks[:, j])
            inter = np.sum(masks[:, i] & masks[:, j])
            jaccard.append(inter / union if union else 0.0)
            abs_r.append(np.abs(np.corrcoef(data[:, i], data[:, j])[0, 1]))
    return {'jaccard': float(np.mean(jaccard)), 'abs_r': float(np.mean(abs_r))}


class Conv3DModel(BaseModel):
    """3D convolutional model designed to decode connectomes (Model 3).
    Extracts features simultaneously along the row fingerprints for each ROI and
    across the frequency channels. Channels should be the last dimension of the
    input data: Shape: ( number of subjects x rows x columns x channels ).

    The model architecture is as follows:
        - Gaussian noise layer (optional, based on stddev hyperparameter)
        - 3D convolutional layer with kernel size (1, n_nodes, n_freqs) to
            convolve across connectivity fingerprints and frequencies.
        - Batch normalization and spatial dropout (optional)
        - Linear convolution (Conv2D with kernel size (n_nodes, 1)) to convolve
            across the remaining spatial dimension).
        - Batch normalization and spatial dropout (optional)
        - Flattening layer
        - Fully connected layer with ReLU activation and L1 regularization
        - Batch normalization and dropout
        - Output layer with linear activation and L1 regularization
    """

    def __init__(self, meta=None, dataset=None, specs_prefix=False):
        """Initialize the 3D convolutional model.

        Parameters
        ----------
        meta : mneflow meta file
        Dataset : mneflow.Dataset
        specs : dict
            Dictionary of model hyperparameters.

            n_latent_dense : int, optional
                Number of latent components in the first dense layer.
                Defaults to 32.
            n_latent : int, optional
                Number of filters in the 3D convolutional layer. Defaults to 32.
            nonlin : callable, optional
                Activation function for the convolutional layers. Defaults to
                `tf.nn.relu`.
            spatial_dropout : float, optional
                Dropout rate for spatial dropout layers after the convolutional
                layers. If set to 0, spatial dropout is not applied after the
                convolutional layers. Defaults to 0.1.
            dropout : float, optional
                Dropout rate  for the fully connected layers. Defaults to 0.4.
            stddev : int, optional
                Standard deviation used for Gaussian layer applied to the inputs.
                If set to 0, Gaussian noise is not applied to the inputs.
                Defaults to 0.1.
        """
        self.scope = 'conv3d'
        meta.model_specs.setdefault('n_latent_dense', 32)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('spatial_dropout', 0.1)
        meta.model_specs.setdefault('dropout', 0.4)
        meta.model_specs.setdefault('stddev', 0.1)
        meta.model_specs.setdefault('stride', 1)
        meta.model_specs.setdefault('nonlin', 'relu')
        meta.model_specs.setdefault('l1_lambda', 0.)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope

        super(Conv3DModel, self).__init__(meta=meta, dataset=dataset,
                                          specs_prefix=specs_prefix)

    def build_graph(self):
        """Build the computational graph for the 3D convolutional model using
        defined placeholder `self.X` as input.

        Returns
        -------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.specs = self.meta.model_specs

        # Gaussian Noise layer
        if self.specs['stddev'] > 0:
            self.gaussian_noise = GaussianNoise(
                stddev=self.specs['stddev'])
            inputs1 = self.gaussian_noise(self.inputs)
            print(f"Gaussian noise with stddev: {self.specs['stddev']} "
                  f"applied to the inputs")
        else:
            inputs1 = self.inputs

        # Expand dimensions to add a channel dimension for Conv3D
        inputs1 = Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs1)
        print("Shape before 3D convolution layer:", inputs1.shape)

        # 3D convolutional layer with kernel size (1, n_nodes, n_freqs) to
        # convolve across connectivity fingerprints and frequencies.
        n_nodes = inputs1.shape[-3]
        n_freqs = inputs1.shape[-2]
        self.col_conv3d = Conv3D(filters=self.specs['n_latent'],
                                 kernel_size=(1, n_nodes, n_freqs),
                                 padding="valid",
                                 data_format=None,
                                 kernel_initializer=initializers.HeUniform(),
                                 bias_initializer=initializers.Constant(0.1),
                                 activation=self.specs['nonlin'])
        rsum = self.col_conv3d(inputs1)
        print(f"Built: 3D layer with {self.specs['n_latent']} filters. "
              f"Input shape before: {inputs1.shape}, shape after: {rsum.shape}")

        # Batch normalization and spatial dropout (optional)
        if self.specs['spatial_dropout'] > 0:
            rsum = BatchNormalization()(rsum)
            rsum = SpatialDropout3D(rate=self.specs['spatial_dropout'],
                                     data_format='channels_last')(rsum)

        # Linear convolution (Conv2D with kernel size (n_nodes, 1)) to convolve
        # across the remaining spatial dimension).
        rsum = Lambda(lambda x: tf.squeeze(x, axis=[2, 3]))(rsum)
        rsum = Lambda(lambda x: tf.expand_dims(x, axis=-1))(rsum)
        self.conv2d = Conv2D(filters=self.specs['n_latent'] // 2,
                             kernel_size=(n_nodes, 1),
                             padding='valid',
                             kernel_initializer=initializers.HeUniform(),
                             bias_initializer=initializers.Constant(0.1),
                             activation='relu')
        out = self.conv2d(rsum)
        print(f"Built: 2D layer with {self.specs['n_latent'] // 2} filters. "
              f"Input shape before: {rsum.shape}, shape after: {out.shape}")

        # Batch normalization and spatial dropout (optional)
        if self.specs['spatial_dropout'] > 0:
            out = BatchNormalization()(out)
            out = SpatialDropout2D(rate=self.specs['spatial_dropout'],
                                   data_format='channels_last')(out)

        # Flattening layer
        flat = Flatten()(out)
        print("Shape after flattening: ", flat.shape)

        # Fully connected layer with ReLU activation and L1 regularization
        self.fc1 = Dense(units=self.specs['n_latent_dense'],
                         activation=tf.nn.relu,
                         kernel_regularizer=keras.regularizers.l1(
                             self.specs['l1_lambda']))
        fc1 = self.fc1(flat)
        print(f"Built: Dense layer with {self.specs['n_latent_dense']} units. "
              f"Input shape before: {flat.shape}, shape after: {fc1.shape}")

        # Batch normalization and dropout
        self.b2n = BatchNormalization(scale=False)
        b_last = self.b2n(fc1)
        dropout = Dropout(self.specs['dropout'], noise_shape=None)(b_last)

        # Output layer with linear activation and L1 regularization
        self.fc2 = Dense(
            units=1,
            activation=tf.identity,
            kernel_regularizer=keras.regularizers.l1(
                self.specs['l1_lambda']))
        y_pred = self.fc2(dropout)

        return y_pred


class WeightedSum3dModel(BaseModel):
    """Weighted sum 3D model designed to decode connectomes (Model 2).

    The model architecture is as follows:
        - Gaussian noise layer (optional, based on stddev hyperparameter)
        - First WeightSum3d layer (depthwise convolution using global row
            filters)
        - Batch normalization and spatial dropout
        - Second WeightSum3d layer (depthwise convolution using global column
            filters, optional based on n_latent2 hyperparameter)
        - Batch normalization and spatial dropout (optional)
        - Optional channel attention block
        - Pointwise convolution (Conv2D with 1x1 kernel)
        - Flattening layer
        - Fully connected layer with ReLU activation and L1 regularization
        - Batch normalization and dropout
        - Output layer with linear activation and L1 regularization"""

    def __init__(self, meta=None, dataset=None, specs_prefix=False):
        """
        Parameters
        ----------
        meta : mneflow meta file
        Dataset : mneflow.Dataset
        specs : dict
           Dictionary of model hyperparameters.

           n_latent_dense : int, optional
               Number of latent components in the first dense layer. Defaults to 32.
           n_latent1 : int, optional
               Number of filters in the first depthwise convolutional layer. Defaults to 8.
           n_latent2 : int, optional
               Number of filters in the second depthwise convolutional layer.
               If set to 0,only the first depthwise layer is applied.
               Defaults to 8.
           n_latent_cross : int, optional
               Number of filters in the pointwise convolutional layer. Defaults to 1.
           nonlin : callable, optional
               Activation function for the convolutional layers. Defaults to `tf.nn.relu`.
           spatial_dropout : float, optional
               Dropout rate for spatial dropout layers after the convoultional layers.
               If set to 0, spatial dropout is not applied after the convolutional layers.
               Defaults to 0.1.
           dropout : float, optional
               Dropout rate  for the fully connected layers. Defaults to 0.4.
           stddev : int, optional
               Standard deviation used for Gaussian layer applied to the inputs.
               If set to 0, Gaussian noise is not applied to the inputs.
               Defaults to 0.05.
           attention_rate: int, optional
               Attention rate for the squeeze-and-excitation channel attention block.
               For details see [1].
               If set to 0, Channel attention is not applied before the pointwise operation.
               Defaults to 0.05.

        Returns
        -------
        None.

        References
        ----------
           [1] Hu, J. et al. (2018). Squeeze-and-excitation networks.
           In Proceedings of the IEEE conference on computer vision and pattern recognition. pages 7132–7141.
        """
        self.scope = 'weighted_sum_3d'

        meta.model_specs.setdefault('n_latent_dense', 32)
        meta.model_specs.setdefault('n_latent_cross', 1)
        meta.model_specs.setdefault('n_latent1', 8)
        meta.model_specs.setdefault('n_latent2', 8)
        meta.model_specs.setdefault('spatial_dropout', 0.1)
        meta.model_specs.setdefault('dropout', 0.4)
        meta.model_specs.setdefault('stddev', 0.05)
        meta.model_specs.setdefault('attention_rate', 4)
        meta.model_specs.setdefault('stride', 1)
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 0.)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])

        meta.model_specs['scope'] = self.scope

        super().__init__(meta=meta, dataset=dataset, specs_prefix=specs_prefix)

    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.specs = self.meta.model_specs

        # Gaussian Noise layer
        if self.specs['stddev'] > 0:
            self.gaussian_noise = GaussianNoise(
                stddev=self.specs['stddev'])
            inputs1 = self.gaussian_noise(self.inputs)
            print(f"Gaussian noise with stddev: {self.specs['stddev']} "
                  f"applied to the inputs")
        else:
            inputs1 = self.inputs

        # Depth-wise convolution using global row filters to compress row dim
        print("Shape before the first depthwise convolution: ", inputs1.shape)
        self.rowsum_3d = WeightSum3d(size=self.specs['n_latent1'],
                                     specs=self.specs,
                                     nonlin=tf.nn.relu)
        rsum = self.rowsum_3d(inputs1)
        print(
            f"Built: Depthwise convolutional layer with {self.specs['n_latent1']} filters. "
            f"Input shape before: {inputs1}, shape after: {rsum}")

        # Batch normalization and spatial dropout (optional)
        if self.specs['spatial_dropout'] > 0:
            rsum = BatchNormalization()(rsum)
            rsum = SpatialDropout2D(rate=self.specs['spatial_dropout'],
                                    data_format='channels_last')(rsum)

        # Second depth-wise convolution using global column filters to compress
        # column dimension (optional based on n_latent2 hyperparameter)
        if self.specs['n_latent2'] > 0:
            self.colsum_3d = WeightSum3d(size=self.specs['n_latent2'],
                                         specs=self.specs,
                                         nonlin=tf.nn.relu)
            csum = self.colsum_3d(rsum)
            print(
                f"Built: Depthwise convolutional layer with {self.specs['n_latent2']} filters. "
                f"Input shape before: {rsum}, shape after: {csum}")

            # Batch normalization and spatial dropout (optional)
            if self.specs['spatial_dropout'] > 0:
                print("Adding batch norm and dropout")
                csum = BatchNormalization()(csum)
                csum = SpatialDropout2D(rate=self.specs['spatial_dropout'],
                                        data_format='channels_last')(csum)
        else:
            print("Using only one depthwise convolutional layer")
            csum = rsum

        # Channel attention block (optional)
        if self.specs['attention_rate'] > 0:
            csum = se_block(csum, ratio=self.specs['attention_rate'])
            print(
                "Built: Squeeze-and-Excitation block added with channel reduction rate: ",
                self.specs['attention_rate'])

        # Pointwise convolution (Conv2D with 1x1 kernel)
        pointwise = Conv2D(self.specs['n_latent_cross'],
                           (1, 1),
                           activation='relu',
                           data_format='channels_last')(csum)
        print(f"Built: Pointwise layer with {self.specs['n_latent_cross']} "
              f"filters. Input shape before: {csum.shape}, shape after: {pointwise.shape}")

        # Flattening layer
        flat = Flatten()(pointwise)
        print("Shape after flattening: ", flat.shape)

        # Fully connected layer with ReLU activation and L1 regularization
        self.fc1 = Dense(units=self.specs['n_latent_dense'],
                         activation=tf.nn.relu,
                         kernel_regularizer=keras.regularizers.l1(
                             self.specs['l1_lambda']))
        fc1 = self.fc1(flat)
        print(f"Built: Dense layer with {self.specs['n_latent_dense']} units. "
              f"Input shape before: {flat.shape}, shape after: {fc1.shape}")

        # Batch normalization and dropout
        self.b2n = BatchNormalization(scale=False)
        b_last = self.b2n(fc1)
        dropout = Dropout(self.specs['dropout'], noise_shape=None)(b_last)

        # Output layer with linear activation and L1 regularization
        self.fc = Dense(units=1,
                        activation=tf.identity,
                        kernel_regularizer=keras.regularizers.l1(
                            self.specs['l1_lambda']))
        y_pred = self.fc(dropout)

        return y_pred


class SymmetricModel(BaseModel):
    """Symmetric 3D model designed to decode connectomes (Model 1).

    Extracts features first along the row fingerprints for each ROI
    and then identifies cross-frequency interactions. Channels should be the
    last dimension of the input data:
    Shape: ( number of subjects x rows x columns x channels )

    The model architecture is as follows:
        - Gaussian noise layer (optional, based on stddev hyperparameter)
        - SquareSum3d layer (symmetric operation across rows and columns
            followed by depthwise convolution)
        - Batch normalization and spatial dropout
        - Optional channel attention block
        - Pointwise convolution (Conv2D with 1x1 kernel)
        - Flattening layer
        - Fully connected layer with ReLU activation and L1 regularization
        - Batch normalization and dropout
        - Output layer with linear activation and L1 regularization
        """

    def __init__(self, meta=None, dataset=None, specs_prefix=False):
        """Initialize the symmetric model.

        Parameters
        ----------
        meta : mneflow meta file
        Dataset : mneflow.Dataset
        specs : dict
           Dictionary of model hyperparameters.

           n_latent_dense : int, optional
               Number of latent components in the first dense layer. Defaults to 32.
           n_latent : int, optional
               Number of filters in the depthwise convolutional layer. Defaults to 8.
           n_latent_cross : int, optional
               Number of filters in the pointwise convolutional layer. Defaults to 1.
           nonlin : callable, optional
               Activation function for the convolutional layers. Defaults to `tf.nn.relu`.
           spatial_dropout : float, optional
               Dropout rate for spatial dropout layers after the convoultional layers.
               If set to 0, spatial dropout is not applied after the convolutional layers.
               Defaults to 0.1.
           dropout : float, optional
               Dropout rate  for the Dense layer. Defaults to 0.4.
           stddev : int, optional
               Standard deviation used for Gaussian layer applied to the inputs.
               If set to 0, Gaussian noise is not applied to the inputs.
               Defaults to 0.05.
           attention_rate: int, optional
               Attention rate for the squeeze-and-excitation channel attention block.
               For details see [1].
               If set to 0, Channel attention is not applied before the pointwise operation.
               Defaults to 0.05.

        Returns
        -------
        None.

        References
        ----------
            [1] Hu, J. et al. (2018). Squeeze-and-excitation networks.
            In Proceedings of the IEEE conference on computer vision and pattern recognition. pages 7132–7141.
        """

        self.scope = 'symmetric_tpk'

        meta.model_specs.setdefault('n_latent_dense', 32)
        meta.model_specs.setdefault('n_latent_cross', 1)
        meta.model_specs.setdefault('n_latent', 8)
        meta.model_specs.setdefault('spatial_dropout', 0.1)
        meta.model_specs.setdefault('dropout', 0.4)
        meta.model_specs.setdefault('stddev', 0.05)
        meta.model_specs.setdefault('attention_rate', 4)
        meta.model_specs.setdefault('stride', 1)
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 0.)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs.setdefault('trainable_pointwise_kernel', True)

        meta.model_specs['scope'] = self.scope

        super(SymmetricModel, self).__init__(meta=meta, dataset=dataset,
                                             specs_prefix=specs_prefix)

    def build_graph(self):
        """
        Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.specs = self.meta.model_specs

        # Gaussian Noise layer
        if self.specs['stddev'] > 0:
            self.gaussian_noise = GaussianNoise(
                stddev=self.specs['stddev'])
            inputs1 = self.gaussian_noise(self.inputs)
            print(f"Gaussian noise with stddev: {self.specs['stddev']} "
                  f"applied to the inputs")
        else:
            inputs1 = self.inputs

        # SquareSum3d layer (symmetric operation across rows and columns
        # followed by deptwise convolution)
        self.squaresum_3d = SquareSum3d(size=self.specs['n_latent'],
                                        specs=self.specs,
                                        nonlin=tf.nn.relu)
        ssum1 = self.squaresum_3d(inputs1)
        print(
            f"Built: Depthwise convolutional layer with {self.specs['n_latent']} "
            f"filters. Input shape before: {inputs1.shape}, shape after: {ssum1.shape}")
        # Intermediate outputs, for the activations (see _activations)
        self._graph_outputs = {'squaresym': ssum1}

        # Batch normalization and spatial dropout (optional). The
        # normalization is per frequency (the last axis).
        self.ssum_bn = None
        if self.specs['spatial_dropout'] > 0:
            self.ssum_bn = BatchNormalization()
            ssum1 = self.ssum_bn(ssum1)
            ssum1 = SpatialDropout2D(rate=self.specs['spatial_dropout'],
                                     data_format='channels_last')(ssum1)

        # Channel attention block (optional): scales each frequency
        if self.specs['attention_rate'] > 0:
            ssum1, se_scale = se_block(ssum1,
                                       ratio=self.specs['attention_rate'],
                                       return_scale=True)
            self._graph_outputs['se_scale'] = se_scale
            print(
                "Built: Squeeze-and-Excitation block added with channel reduction rate: ",
                self.specs['attention_rate'])

        # Pointwise convolution (Conv2D with 1x1 kernel)
        if self.specs['trainable_pointwise_kernel']:
            # Each component (axis 1) gets its own weighting of the last
            # axis. Applied channels_last (see _apply_pointwise): TensorFlow
            # has no channels_first depthwise convolution on the CPU.
            self.pointwise = DepthwiseConv2D(
                kernel_size=(1, self.meta.data['n_ch']),
                activation='relu',
                padding="valid",
                data_format='channels_last',
                depth_multiplier=1)
        else:
            self.pointwise = Conv2D(self.specs['n_latent_cross'], (1, 1),
                                    activation='relu',
                                    data_format='channels_last')

        pointwise = self._apply_pointwise(ssum1)
        print(f"Built: Pointwise layer with {self.specs['n_latent_cross']} "
              f"filters. Input shape before: {ssum1.shape}, shape after: {pointwise.shape}")
        self._graph_outputs['pointwise'] = pointwise

        # Flattening layer
        self.flat = Flatten()
        flat = self.flat(pointwise)
        print("Flattened shape: ", flat.shape)

        # Fully connected layer with ReLU activation and L1 regularization
        self.fc1 = Dense(
            units=self.specs['n_latent_dense'],
            activation=tf.nn.relu,
            kernel_regularizer=keras.regularizers.l1(
                self.specs['l1_lambda']))
        fc1 = self.fc1(flat)
        self._graph_outputs['fc1'] = fc1

        print(f"Built: Dense layer with {self.specs['n_latent_dense']} units. "
              f"Input shape before: {flat.shape}, shape after: {fc1.shape}")

        # Batch normalization and dropout
        self.b2n = BatchNormalization(scale=False)
        batch = self.b2n(fc1)
        dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(batch)

        # Output layer with linear activation and L1 regularization
        self.fc = Dense(
            units=1,
            activation=tf.identity,
            kernel_regularizer=keras.regularizers.l1(
                self.specs['l1_lambda']))
        y_pred = self.fc(dropout)
        self._graph_outputs['fin_fc'] = y_pred
        self._activation_model = None

        return y_pred

    def _activations(self, X):
        """The model's activations for the inputs X, in inference mode.

        Returns a dict of numpy arrays: ``'squaresym'`` (the
        SquareSum3d output), ``'se_scale'`` (the squeeze-and-excitation
        scale of each frequency, shape (n_samples, n_freqs); only with
        attention), ``'pointwise'``, ``'fc1'`` and ``'fin_fc'`` (the
        output). The layers in between (batch normalization, dropout,
        attention) are applied as in the model.
        """
        if self._activation_model is None:
            self._activation_model = keras.Model(self.inputs,
                                                 self._graph_outputs)
        out = self._activation_model(X, training=False)
        return {k: np.asarray(v) for k, v in out.items()}

    def _effective_pointwise(self, se_scale=None):
        """The pointwise weights of the SquareSum3d output.

        Batch normalization and the attention block scale each
        frequency of the SquareSum3d output before the pointwise layer:
        frequency f's weight is ``kernel[f] * gamma_f /
        sqrt(var_f + eps) * mean(se_scale[:, f])``. The normalization's
        offsets don't depend on the input, and the attention scale is
        averaged over the samples. Unlike the kernel, these weights act
        on the (non-negative) SquareSum3d output.

        Parameters
        ----------
        se_scale : array-like, shape (n_samples, n_freqs), optional
            The attention scales of some data (see
            :meth:`_activations`). Required if the model has an
            attention block.

        Returns
        -------
        weights : ndarray
            The shape of ``extract_weights()['pointwise']``: frequency
            first.
        """
        kernel = np.squeeze(self.pointwise.kernel.numpy())
        gain = np.ones(kernel.shape[0])
        if self.ssum_bn is not None:
            bn = self.ssum_bn
            gamma = bn.gamma.numpy() if bn.scale else 1.
            gain = gain * gamma / np.sqrt(bn.moving_variance.numpy()
                                          + bn.epsilon)
        if 'se_scale' in self._graph_outputs:
            if se_scale is None:
                raise ValueError("The attention scales (se_scale) are "
                                 "required: the model has attention.")
            gain = gain * np.mean(se_scale, axis=0)
        return kernel * gain.reshape(-1, *[1] * (kernel.ndim - 1))

    def _standardized_pointwise(self, activations):
        """The pointwise weights per standard deviation of each frequency's
        SquareSum3d feature.

        Batch normalization divides frequency f of the SquareSum3d output by
        its standard deviation sigma_f, so the model doesn't depend on the
        scale of each frequency's features, and the effective weights
        (:meth:`_effective_pointwise`, which include 1/sigma_f) rank the
        frequencies partly by that arbitrary scale. These weights leave
        out 1/sigma_f: ``kernel[f] * gamma_f * mean(se_scale[:, f])``,
        the change of the pointwise input per standard deviation of the
        feature (a standardized coefficient). Without batch normalization,
        gamma_f / sigma_f is replaced by 1 and the kernel multiplied by the
        features' standard deviation.

        Parameters
        ----------
        activations : dict
            :meth:`_activations` of some data (``'squaresym'``, and
            ``'se_scale'`` with attention).

        Returns
        -------
        weights : ndarray
            The shape of ``extract_weights()['pointwise']``: frequency
            first.
        """
        kernel = np.squeeze(self.pointwise.kernel.numpy())
        if self.ssum_bn is not None:
            bn = self.ssum_bn
            gain = (bn.gamma.numpy() if bn.scale
                    else np.ones(kernel.shape[0]))
        else:
            gain = np.std(activations['squaresym'], axis=(0, 1, 2))
        if 'se_scale' in self._graph_outputs:
            gain = gain * np.mean(activations['se_scale'], axis=0)
        return kernel * gain.reshape(-1, *[1] * (kernel.ndim - 1))

    def _apply_pointwise(self, x) -> tf.Tensor:
        """Apply the pointwise layer to the SquareSum3d output ``x``.

        With ``trainable_pointwise_kernel``, the depthwise convolution
        treats axis 1 of ``x`` as its channels (one kernel over the last
        axis per component). It runs channels_last, so ``x`` is transposed
        to put axis 1 last, and the output transposed back.
        """
        if isinstance(self.pointwise, DepthwiseConv2D):
            x = keras.ops.transpose(x, (0, 2, 3, 1))
            # cast: keras.ops is typed for all backends
            return cast(tf.Tensor,
                        keras.ops.transpose(self.pointwise(x), (0, 3, 1, 2)))
        return self.pointwise(x)

    def extract_weights(self, verbose=False):
        """Extract the trained weights relevant for interpretation.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the shapes of the extracted weights.
            Defaults to False.

        Returns
        -------
        weights : dict
            Dictionary with keys ``'ssum1'`` and ``'ssum1b'`` (the
            `SquareSum3d` layer's weights and bias), ``'pointwise'`` (the
            pointwise convolution kernel), ``'fc1_w_flat'`` (the first
            dense layer's weights, if present) and ``'out_w_flat'`` (the
            output layer's weights).

        """
        weights = {}

        # Extract weights

        # Connectivity fingerprint extraction fiters
        weights['ssum1'] = np.squeeze(self.squaresum_3d.w.numpy())
        weights['ssum1b'] = self.squaresum_3d.b_in.numpy()

        # Pointwise Convolution kernels
        weights['pointwise'] = np.squeeze(self.pointwise.kernel.numpy())

        # First dense layer
        if hasattr(self, "fc1"):
            weights['fc1_w_flat'] = self.fc1.kernel.numpy()

        # Final layer
        weights['out_w_flat'] = self.fc.kernel.numpy()

        if verbose:
            fc1_shape = (weights['fc1_w_flat'].shape
                         if 'fc1_w_flat' in weights else None)
            print("""Weights: \n
                  SPATIAL: {}
                  POINTWISE: {}
                  FC1: {}
                  FC_OUT: {}""".format(weights['ssum1'].shape,
                  weights['pointwise'].shape,
                  fc1_shape,
                  weights['out_w_flat'].shape))

        return weights

    def compute_patterns(self, data_path=None, verbose=False, shapley_order=1,
                         methods=['weight', 'compwise_loss', 'output_corr']):
        """Extracts weights, activation patterns, validation data covariances,
        and feature relevances for a single training fold.
        Required for visualization.

        Parameters
        ----------
        data_path : str, list of str, or tf.data.Dataset, optional
            Path(s) to TFRecord files (or an already-built
            ``tf.data.Dataset``) on which the patterns are estimated.
            Defaults to None, in which case the model's own
            validation dataset (``self.dataset.val``) is used.

        verbose : bool, optional
            Whether to print the shapes of the computed activations.
            Defaults to False.

        shapley_order : int, optional
            Currently unused. Defaults to 1.

        methods : list of str, optional
            Which interpretation methods to additionally compute.
            Only ``'output_corr'`` has an effect (adds
            ``corr_to_output`` via :meth:`get_output_correlations`).
            Defaults to ``['weight', 'compwise_loss', 'output_corr']``.

        Returns
        -------
        patterns_struct : dict
            Dictionary with keys
                ``'weights'`` model weights, see:meth:`extract_weights`,
                and ``'pointwise_effective'`` (the pointwise weights of
                the SquareSum3d output, see :meth:`_effective_pointwise`),
                ``'ccms'`` mean activation of each layer, per class, over one batch, 
                ``'dcov'`` sample covariance of the input, 
                ``'covs'`` covariance of selected layer activations, 
                ``'patterns'`` and ``'spectra'`` (currently unused, left empty), 
                ``'freqs'`` (currently unused, left None), and, if 
                ``'output_corr'`` is in ``methods``, ``'corr_to_output'`` (see
                :meth:`get_output_correlations`).

        Raises
        ------
        AttributeError
            If ``data_path`` is not a string, list/tuple of strings,
            or ``tf.data.Dataset``.

        """
        patterns_struct = {'weights' : {'squaresym':[], 'pointwise':[],
                                        'fin_fc':[], 'fc1':[]},
                           'ccms' : {'squaresym':[], 'pointwise':[], 'fin_fc':[],
                                     'fc1':[],
                                     'input':[]},
                           'dcov' : {'input_spatial':[], 'class_conditional':[],
                                     'k-1':[]},
                           'covs' : {'fc':[], 'fc1':[], 'squaresym':[]},
                           'patterns' : {},
                           'spectra': {},
                           'freqs': None
                           }

        if not data_path:
            print("Computing patterns: No path specified, using validation dataset (Default)")
            ds = self.dataset.val
        elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):

            ds = self.dataset._build_dataset(data_path,
                                             split=False,
                                             test_batch=None,
                                             repeat=True)

        elif isinstance(data_path, tf.data.Dataset):
            ds = data_path
        else:
            raise AttributeError('Specify dataset or data path.')


        X, y = [row for row in ds.take(1)][0]

        # Layer activations, through the model's layers (incl. the batch
        # normalization and attention between SquareSum3d and the
        # pointwise layer). (Were computed without them.)
        activations = self._activations(X)

        if verbose:
            print(""""Activations: \n
                  SPATIAL: {}
                  POINTWISE: {}
                  FC1: {}
                  FC_OUT: {}""".format(
                  activations['squaresym'].shape,
                  activations['pointwise'].shape,
                  activations['fc1'].shape,
                  activations['fin_fc'].shape))

        patterns_struct['ccms']['squaresym'] = np.mean(activations['squaresym'], 0)
        patterns_struct['ccms']['pointwise'] = np.mean(activations['pointwise'], 0)
        if hasattr(self, "fc1"):
            patterns_struct['ccms']['fc1'] = np.mean(activations['fc1'], 0)
        patterns_struct['ccms']['fin_fc'] = np.mean(activations['fin_fc'], 0)
        patterns_struct['ccms']['input'] = np.mean(X, 0)

        n_samp = np.shape(X)[0]
        patterns_struct['covs']['fin_fc'] = np.cov(activations['fc1'].T)
        fc1_flat = np.reshape(activations['pointwise'], [n_samp, -1])
        patterns_struct['covs']['fc1'] = np.cov(fc1_flat.T)

        weights = self.extract_weights()
        # The pointwise weights of the SquareSum3d output, with this
        # data's attention scales (for the frequency patterns)
        weights['pointwise_effective'] = self._effective_pointwise(
            activations.get('se_scale'))

        # Spatial covariance of the input rows, for the activation patterns.
        # (Was np.einsum('hijk, hilk -> ilk'), which correlates each row's
        # sum with its own entries and is not a covariance.)
        dcov = input_covariance(X)


        patterns_struct['weights'] = weights
        patterns_struct['dcov'] = dcov

        if 'output_corr' in methods:
            patterns_struct['corr_to_output'] = self.get_output_correlations(activations, y)

        del X, activations

        return patterns_struct

    def get_roi_labels(self, percentile=80, patterns=None,
                       methods=['row', 'col', 'diag'], selection_method='activation',
                       return_stability=False):
        """
        Computes ROIs from spatial activation patterns.


        Parameters
        ----------
        percentile : int, optional
            Inclusion threshold top (100 - percetile)% active sources across
            all folds based on selection_method. Default 80

        patterns : dict, optional
            Output of self.extract_patterns, if not provided computed.
            Default None.

        methods : list of str, optional
            Which spatial pattern keys (``patterns['spatial_patterns']``)
            to compute ROI labels for. Defaults to
            ``['row', 'col', 'diag']``.

        selection_method : str, optional
            How to integrate across folds. Possible arguments 'count' or
            'activation'. Default 'activation'

        return_stability : bool, optional
            Whether to also return how consistent the patterns are across
            folds. Defaults to False.

        Returns
        ------
        roi_labels : dict
            ROI indices identified by the interpretation pipeline
            {'extraction_method':[label indices]}

        stability : dict
            Only if ``return_stability`` is True. For each method, a dict
            with the mean over all pairs of folds of
            ``'jaccard'``, the Jaccard index of the sets of top sources
            (sources whose absolute standardized pattern exceeds the
            selection threshold), and ``'abs_r'``, the absolute correlation
            of the standardized patterns.
        """

        if not patterns:
            patterns = self.extract_patterns()

        roi_labels = {}
        stability = {}

        for key in methods:
            data = (patterns['spatial_patterns'][key] - patterns['spatial_patterns'][key].mean(0)) / patterns['spatial_patterns'][key].std(0, keepdims=True)
            if selection_method == 'count':
                top_active = np.percentile(data, percentile, axis=0)
                top_label_inds = []
                for f in range(data.shape[-1]):
                    top_label_inds.append(data[:, f] > top_active[f])
                counts = np.sum(np.stack(top_label_inds, 0), 0)
                mask = np.where(counts >= 4)[0]
            elif selection_method == 'activation':
                mean_activation = np.mean(data, -1)
                top_active = np.percentile(mean_activation, percentile)
                mask = np.where(mean_activation > top_active)[0]
            else:
                raise ValueError(f"selection_method must be 'count' or "
                                 f"'activation', got {selection_method!r}")

            roi_labels[key] = mask
            if return_stability:
                stability[key] = _fold_stability(data, top_active)
        if 'row' in methods and 'col' in methods:
            roi_labels['row-col'] = np.unique(np.concatenate([roi_labels['row'],
                                                              roi_labels['col']]))

        if return_stability:
            return roi_labels, stability
        return roi_labels

    # def get_frequency_kernels(self, methods=['row', 'col', 'diag'], patterns=None):
    #     if not patterns:
    #         patterns = self.extract_patterns()

    #     freq_weights = {}
    #     activations = patterns_struct['ccms']['pointwise']

    #     for key in methods:
    #         data = (patterns['spatial_patterns'][key] - patterns['spatial_patterns'][key].mean(0)) / patterns['spatial_patterns'][key].std(0, keepdims=True)
    #         if selection_method == 'count':
    #             top_active = np.percentile(data, percentile, axis=0)
    #             top_label_inds = []
    #             for f in range(data.shape[-1]):
    #                 top_label_inds.append(data[:, f] > top_active[f])
    #             counts = np.sum(np.stack(top_label_inds, 0), 0)
    #             mask = np.where(counts >= 4)[0]
    #         elif selection_method == 'activation':
    #             mean_activation = np.mean(data, -1)
    #             top_active = np.percentile(mean_activation, percentile)
    #             mask = np.where(mean_activation > top_active)[0]

    #         roi_labels[key] = mask
    #     if 'row' in methods and 'col' in methods:
    #         roi_labels['row-col'] = np.unique(np.concatenate([roi_labels['row'],
    #                                                           roi_labels['col']]))


    #     return

    def ablation_analysis(self, name, label_inds, hyperparameters):
        """Build a new SymmetricModel restricted to a subset of samples/labels.

        Deep-copies this model's metadata, restricts it to
        ``label_inds`` via ``meta.data['sample_subset']``, gives it a
        new ``data_id`` tagged with ``name``, and builds a new
        :class:`Dataset` and :class:`SymmetricModel` from it, reusing
        this model's ``model_specs``. Used to evaluate the effect of
        ablating (restricting to) a subset of ROIs/labels identified
        by an interpretation method.

        Parameters
        ----------
        name : 'str'
            Name of the pattern extraction method.
        label_inds : list of int
            Label Indices
        hyperparameters : dict
            Hyperparameters for the ablation model. Must contain
            ``'batch_size'``.

        Returns
        -------
        meta_abl : mneflow.MetaData
            Metadata for the ablation dataset/model, with
            ``sample_subset``, ``data_id``, and ``model_specs`` set.
        model_abl : SymmetricModel
            A newly constructed (unbuilt) SymmetricModel using
            ``meta_abl`` and ``dataset``.
        dataset : mneflow.Dataset
            Dataset restricted to ``label_inds``.

        """
        meta_abl = deepcopy(self.meta)
        meta_abl.data['sample_subset'] = label_inds
        meta_abl.data['data_id'] = "_".join([meta_abl.data['data_id'], 'abl_activation_', name])

        #meta_abl.data['n_seq'] = len(label_inds)
        #meta_abl.data['n_t'] = len(label_inds)

        dataset = Dataset(meta_abl, train_batch=hyperparameters['batch_size'],
                                       sample_subset=label_inds)
        meta_abl.update(model_specs=self.meta.model_specs)
        # The weights and patterns of this model don't belong to the
        # ablation model (and the patterns are large)
        meta_abl.weights = {}
        meta_abl.patterns = {}

        model_abl = SymmetricModel(meta=meta_abl,
                                   dataset=dataset,
                                   specs_prefix=False)

        return meta_abl, model_abl, dataset



    def get_output_correlations(self, activations, y_true):
        """Computes a similarity metric between each of the extracted
        features and the target variable.

        The metric is the Pearson correlation for discrete
        (``target_type == 'int'``) targets, and the Spearman
        correlation for continuous (``target_type in ['float',
        'signal']``) targets.

        Parameters
        ----------
        activations : dict
            Layer activations as returned within
            :meth:`compute_patterns`; only ``activations['fc1']`` is
            used here.

        y_true : tf.Tensor, shape (n_samples, y_shape)
            Target values for the same batch as ``activations``.

        Returns
        -------
        corr_to_output : ndarray, shape (y_shape, n_features)
            Per-target, per-feature correlation between the flattened
            ``'fc1'`` activations and ``y_true``, with any NaNs (e.g.
            from constant features) replaced by 0.

        """
        corr_to_output = []
        y_true = y_true.numpy()
        flat_feats = self.flat(activations['fc1']).numpy()#.reshape(y_true.shape[0], -1)


        for y_ in y_true.T:
            if self.dataset.h_params['target_type'] in ['float', 'signal']:
                rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])



            elif self.dataset.h_params['target_type'] == 'int':
                rfocs = np.array([pearsonr(y_, f)[0] for f in flat_feats.T])


            corr_to_output.append(rfocs)

        corr_to_output = np.concatenate(corr_to_output, -1)

        if np.any(np.isnan(corr_to_output)):
            corr_to_output[np.isnan(corr_to_output)] = 0
        return corr_to_output

    def collect_patterns(self, fold=0, n_folds=1, n_comp=1,
                          methods=['weight',
                                  #'compwise_loss',
                                  'output_corr'
                                  ]):
        """Collects patterns computed for each fold during cross-validation.

        Calls :meth:`compute_patterns` for the current fold and
        accumulates its results, in place, into ``self.cv_patterns``
        (keyed by ``'dcov'``, ``'ccms'``, and, per method in
        ``methods``, ``'feature_relevance'``) and ``self.cv_weights``.
        On the first fold (``fold == 0`` or ``self.cv_patterns`` still
        empty), these arrays are (re-)allocated with a trailing
        ``n_folds`` axis.

        Parameters
        ----------
        fold : int, optional
            Index of the current cross-validation fold, used to
            select the slice of ``self.cv_patterns``/``self.cv_weights``
            to write into. Defaults to 0.

        n_folds : int, optional
            Not used; the number of CV folds is read from
            ``self.meta.data['folds']`` on the first fold.
            Defaults to 1.

        n_comp : int, optional
            Currently unused. Defaults to 1.

        methods : list of str, optional
            Which feature-relevance keys of ``self.cv_patterns`` to
            allocate/update. Defaults to ``['weight', 'output_corr']``.

        Returns
        -------
        None

        """
        patterns_struct = self.compute_patterns()

        if len(self.cv_patterns.items()) == 0 or fold==0:
            n_parcels = self.meta.data['n_seq']
            n_fft = self.meta.data['n_ch']
            # Number of CV folds; meta.data['n_folds'] also counts the
            # holdout fold when test_set='holdout'
            n_folds = len(self.meta.data['folds'][0])

            #self.cv_patterns['freqs'] = patterns_struct['freqs']

            self.cv_patterns['dcov'] = np.zeros([n_parcels, n_parcels, n_fft, n_folds])
            #Feature relevances
            for method in methods:
                self.cv_patterns[method]['feature_relevance'] = np.zeros([
                                                            *self.flat.input.shape[1:],
                                                            n_folds])
            for k in patterns_struct['ccms'].keys():
                shape = [*patterns_struct['ccms'][k].shape, n_folds]

                self.cv_patterns['ccms'][k] = np.zeros(shape)

        max_weight = np.argmax(patterns_struct['weights']['out_w_flat'], 0)
        weight_relevance = patterns_struct['weights']['fc1_w_flat'][:, max_weight].reshape(self.flat.input.shape[1:])

        self.cv_patterns['weight']['feature_relevance'][:, :, :, fold] = weight_relevance
        self.cv_patterns['dcov'][:, :, :, fold] = patterns_struct['dcov']

        max_corr = np.argmax(patterns_struct['corr_to_output'], 0)
        corr_relevance = patterns_struct['weights']['fc1_w_flat'][:, max_corr].reshape(self.flat.input.shape[1:])
        self.cv_patterns['output_corr']['feature_relevance'][:, :, :, fold] = corr_relevance

        for k in patterns_struct['weights'].keys():
            self.cv_weights[k].append(patterns_struct['weights'][k])

        for k in patterns_struct['ccms'].keys():
            self.cv_patterns['ccms'][k][..., fold] = patterns_struct['ccms'][k]




    def _validation_data(self, fold):
        """Inputs of one CV fold's validation set (its first batch: the
        whole fold, see :meth:`recompute_input_covariance`)."""
        _, val = self.dataset._build_dataset(
            self.dataset.h_params['train_paths'],
            train_batch=self.dataset.training_batch,
            test_batch=None, split=True, val_fold_ind=fold)
        X, _ = next(iter(val))
        return X

    def recompute_input_covariance(self):
        """Recompute ``self.cv_patterns['dcov']`` for every CV fold from
        that fold's validation data, with :func:`input_covariance`.

        The covariance depends only on the data, so it can be recomputed
        for a trained model, e.g. one trained before the covariance and
        the per-fold validation data were fixed (the covariance used to be
        computed with the wrong formula, from the first fold's validation
        data in every fold). (Such a model with batch normalization or
        attention still needs retraining for :meth:`extract_patterns`.)
        Uses the whole validation fold. Training uses the first batch of
        the validation fold (the Dataset's validation batch size, its
        ``test_batch``, or ``train_batch`` if not given): the same data if
        the batch is at least as large as the fold (e.g. batches of 50,
        folds of 48).
        """
        n_folds = len(self.dataset.h_params['folds'][0])
        dcov = None
        for fold in range(n_folds):
            fold_cov = input_covariance(self._validation_data(fold))
            if dcov is None:
                dcov = np.zeros(fold_cov.shape + (n_folds,))
            dcov[..., fold] = fold_cov
        self.cv_patterns['dcov'] = dcov
        return dcov

    def extract_patterns(self, relevance=None, standardized=None):
        """
        Extracts spatial connectivity patterns from a cross-validated model.

        For each fold, identifies the row, column, and diagonal
        components of the flattened pointwise-layer output that most
        strongly drive the output (via
        ``self.meta.weights['fc1_w_flat']`` and
        ``self.meta.weights['out_w_flat']``), then reconstructs each
        component's spatial connectivity pattern from
        ``self.cv_patterns['dcov']`` and the corresponding
        :class:`~mneflow.layers.SquareSum3d` spatial weight
        (``self.meta.weights['ssum1']``).

        The pointwise layer weights the feature of the pair (row, col)
        with the row component's frequency weights. So the row and col
        components take the frequency weights of their pair, and the same
        frequency. The frequency weights are those of the SquareSum3d
        output (``self.meta.weights['pointwise_effective']``, see
        :meth:`_effective_pointwise`): the pointwise kernel scaled by the
        batch normalization and attention in between.

        With ``relevance``, the components and frequencies are chosen by
        the relevance of the SquareSum3d features instead (gradient x
        input, :meth:`feature_relevance`): the pair (row, col) with the
        largest relevance summed over the frequencies, at its most
        relevant frequency; 'diag' the most relevant pair (a, a), at its
        most relevant frequency. The weights can't rank the frequencies:
        the batch normalization after SquareSum3d makes the output
        independent of each frequency's scale, which comes from the norm
        of the spatial filters, so the effective weights rank the
        frequencies by the inverse of that scale.

        With ``standardized``, the weights choose as without, but per
        standard deviation of what they act on, so that the arbitrary
        scale of the features doesn't enter: the frequency by the
        standardized pointwise weights (:meth:`_standardized_pointwise`,
        without the batch normalization's 1/sigma_f), and the pair by the
        dense weights times the standard deviation of each pair's
        pointwise output.

        Parameters
        ----------
        relevance : ndarray, shape (n_latent, n_latent, n_freqs, n_folds), optional
            The relevance of each fold's features, e.g.
            ``extract_combined_patterns()['relevance']``. If None, the
            weights choose (as before).
        standardized : dict, optional
            ``'pointwise'`` (the standardized pointwise weights, shape of
            the effective ones with the folds last) and ``'pair_sd'``
            (n_latent * n_latent, n_folds; the standard deviation of each
            pair's pointwise output, in the order of the dense layer's
            input), e.g. from :meth:`extract_combined_patterns`. Not with
            ``relevance``.

        Returns
        -------
        out : dict
            Nested dict with keys
            ``'spatial_patterns'``,
            ``'freq_patterns'``,
            ``'freq_inds'``, and
            ``'comp_inds'``,
            each itself a dict with sub-keys ``'row'``, ``'col'``, and ``'diag'``.
            ``spatial_patterns[key]`` has shape ``(n_parcels, n_folds)``:
            ``dcov[:, :, f] @ w``, the activation pattern of the
            component's spatial weights w at its frequency f.
            ``freq_patterns[key]`` has shape ``(n_freqs, n_folds)``: the
            frequency weights of the component's feature (the same for
            'row' and 'col'), or with ``relevance`` the feature's
            relevance at each frequency; ``freq_inds[key]`` (the frequency
            f, the largest weight, for 'diag' the largest absolute weight;
            or the largest relevance) and ``comp_inds[key]`` have shape
            ``(n_folds,)``.

        Raises
        ------
        ValueError
            If the model has batch normalization or attention before the
            pointwise layer, and its metadata has no
            ``'pointwise_effective'`` weights (trained with an older
            mneflow): retrain it.

        """
        out = defaultdict(dict)
        # One weight set per trained fold. meta.data['n_folds'] also counts
        # the holdout fold when test_set='holdout', so it can't be used here.
        n_folds = self.meta.weights['out_w_flat'].shape[-1]
        if relevance is not None and standardized is not None:
            raise ValueError("Give relevance or standardized, not both")
        if relevance is not None:
            relevance = np.asarray(relevance)
            if relevance.ndim != 4 or relevance.shape[-1] != n_folds:
                raise ValueError(
                    "relevance must have shape (n_latent, n_latent, "
                    f"n_freqs, {n_folds}), got {relevance.shape}")
        max_inds_folds = [np.argmax(self.meta.weights['out_w_flat'][:, :, i], 0)
                          for i in range(n_folds)]
        pointwise_shape = self.cv_patterns['ccms']['pointwise'].shape[:-1]
        connectivity_patterns = defaultdict(list)
        freq_inds = defaultdict(list)
        comp_inds = defaultdict(list)
        freq_patterns = defaultdict(list)

        pair_sd = None
        if standardized is not None:
            pointwise = np.asarray(standardized['pointwise'])
            pair_sd = np.asarray(standardized['pair_sd'])
            if pointwise.shape[-1] != n_folds or pair_sd.shape[-1] != n_folds:
                raise ValueError("standardized weights must have one set "
                                 f"per fold ({n_folds})")
        elif 'pointwise_effective' in self.meta.weights:
            pointwise = self.meta.weights['pointwise_effective']
        elif self.ssum_bn is not None or 'se_scale' in self._graph_outputs:
            raise ValueError(
                "No 'pointwise_effective' weights in the metadata: the "
                "model was trained with an mneflow version that didn't "
                "store them. Retrain it to extract its patterns.")
        else:
            # Nothing between SquareSum3d and the pointwise layer
            pointwise = self.meta.weights['pointwise']
        trainable_pointwise = self.meta.model_specs.get(
            'trainable_pointwise_kernel', True)
        if not trainable_pointwise and pointwise.ndim != 2:
            raise NotImplementedError(
                "extract_patterns supports trainable_pointwise_kernel=False "
                "only with n_latent_cross=1")

        def freq_weights(comp, fold):
            # With the trainable pointwise layer each component has its own
            # frequency weights, shape (n_freqs, n_components, n_folds).
            # Otherwise a 1x1 convolution weights the frequencies, shared by
            # all components: shape (n_freqs, n_folds).
            if trainable_pointwise:
                return pointwise[:, comp, fold]
            return pointwise[:, fold]

        for i, ind in enumerate(max_inds_folds):
            if relevance is None:
                fc1_w = self.meta.weights['fc1_w_flat'][:, ind, i]
                if pair_sd is not None:
                    # Per standard deviation of each pair's value
                    fc1_w = fc1_w * pair_sd[:, i].reshape(
                        -1, *[1] * (fc1_w.ndim - 1))
                fc1 = np.dot(
                    fc1_w[..., None],
                    self.meta.weights['out_w_flat'][ind, :, i])
                fcr = np.reshape(fc1, pointwise_shape[:-1])

                # Row and column of the strongest component pair. (The row
                # used to come from the argmax of the diagonal, an index
                # 0..n-1 that np.unravel_index mapped to row 0 in every
                # fold.)
                row, col = np.unravel_index(np.argmax(fcr), fcr.shape)

                # The pair's feature is weighted by the row component's
                # frequency weights: both components take its frequency.
                # (The col component used to take its own weights, which
                # weight the features (col, *), not this pair.)
                pair_weights = freq_weights(row, i)
                max_freq_row = max_freq_col = np.argmax(pair_weights)
                diag = np.argmax(np.abs(np.diag(fcr)))
                diag_weights = freq_weights(diag, i)
                max_freq_diag = np.argmax(np.abs(diag_weights))
            else:
                R = relevance[..., i]
                pair_relevance = R.sum(axis=-1)
                row, col = np.unravel_index(np.argmax(pair_relevance),
                                            pair_relevance.shape)
                pair_weights = R[row, col]
                max_freq_row = max_freq_col = np.argmax(pair_weights)
                diag = np.argmax(np.diag(pair_relevance))
                diag_weights = R[diag, diag]
                max_freq_diag = np.argmax(diag_weights)
            spatial_kernel_row = self.meta.weights['ssum1'][max_freq_row, :, row, i]
            spatial_pattern_row = np.dot(self.cv_patterns['dcov'][:, :, max_freq_row, i], spatial_kernel_row.T)

            spatial_kernel_col = self.meta.weights['ssum1'][max_freq_col, :, col, i]
            spatial_pattern_col = np.dot(self.cv_patterns['dcov'][:, :, max_freq_col, i], spatial_kernel_col.T)


            spatial_kernel_diag = self.meta.weights['ssum1'][max_freq_diag, :, diag, i]
            spatial_pattern_diag = np.dot(self.cv_patterns['dcov'][:, :, max_freq_diag, i], spatial_kernel_diag.T)

            comp_inds['row'].append(row)
            comp_inds['col'].append(col)
            comp_inds['diag'].append(diag)

            freq_inds['row'].append(max_freq_row)
            freq_inds['col'].append(max_freq_col)
            freq_inds['diag'].append(max_freq_diag)

            connectivity_patterns['row'].append(spatial_pattern_row)
            connectivity_patterns['col'].append(spatial_pattern_col)
            connectivity_patterns['diag'].append(spatial_pattern_diag)

            freq_patterns['row'].append(pair_weights)
            freq_patterns['col'].append(pair_weights)
            freq_patterns['diag'].append(diag_weights)

        out['spatial_patterns']['row'] = np.stack(connectivity_patterns['row'], -1)
        out['spatial_patterns']['col'] = np.stack(connectivity_patterns['col'], -1)
        out['spatial_patterns']['diag'] = np.stack(connectivity_patterns['diag'], -1)
        out['freq_inds']['row'] = np.array(freq_inds['row'])
        out['freq_inds']['col'] = np.array(freq_inds['col'])
        out['freq_inds']['diag'] = np.array(freq_inds['diag'])
        out['comp_inds']['row'] = np.array(comp_inds['row'])
        out['comp_inds']['col'] = np.array(comp_inds['col'])
        out['comp_inds']['diag'] = np.array(comp_inds['diag'])
        for key in ('row', 'col', 'diag'):
            out['freq_patterns'][key] = np.stack(freq_patterns[key], -1)

        return out

    def feature_relevance(self, X):
        """Relevance of each SquareSum3d feature for the prediction.

        Gradient x input at the SquareSum3d output ``S`` (after its
        ReLU): ``R[a, b, f] = mean_h |dy_h / dS[h, a, b, f] *
        S[h, a, b, f]|`` over the samples h of X, in inference mode.
        Unlike the weights, it accounts for everything between the
        features and the output (batch normalization, attention, the
        pointwise and dense layers and their ReLUs).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_nodes, n_nodes, n_freqs)

        Returns
        -------
        R : ndarray, shape (n_latent, n_latent, n_freqs)
        """
        if self._activation_model is None:
            self._activation_model = keras.Model(self.inputs,
                                                 self._graph_outputs)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape() as tape:
            out = self._activation_model(X, training=False)
            features = out['squaresym']
        # features depend on the (watched) trainable weights, so the tape
        # recorded the path from them to the output. The samples are
        # independent (inference mode), so the gradient of the summed
        # outputs is each sample's gradient
        grad = tape.gradient(out['fin_fc'], features)
        return np.mean(np.abs(grad.numpy() * features.numpy()), axis=0)

    def combined_pattern(self, X, dcov):
        """Activation pattern combined over all spatial filters and
        frequencies, weighted by their relevance (current weights).

        For spatial filter a at frequency f, the activation pattern is
        ``p[:, a, f] = dcov[:, :, f] @ w[f, :, a]`` (as in
        :meth:`extract_patterns`). Its relevance is that of the features
        it takes part in, as row or column (``S[a, b, f] = w_a^T X_f
        w_b``): ``r[a, f] = sum_b R[a, b, f] + R[b, a, f]``, with R from
        :meth:`feature_relevance` on X. The combined pattern is
        ``sum_{a, f} r[a, f] |z(p[:, a, f])| / sum r``, z the z-score
        over the parcels: the absolute value as the sign of a pattern is
        arbitrary (w^T X w), and the z-score so that filters with a large
        covariance don't dominate.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_nodes, n_nodes, n_freqs)
            The data for the relevance, e.g. the model's validation fold.
        dcov : ndarray, shape (n_nodes, n_nodes, n_freqs)
            The input covariance (:func:`input_covariance`), of the same
            data.

        Returns
        -------
        out : dict
            ``'pattern'`` (n_nodes,), the combined pattern;
            ``'freq_relevance'`` (n_freqs,) and ``'filter_relevance'``
            (n_latent, n_freqs), the relevance of each frequency and of
            each filter at each frequency, as shares of the total; and
            ``'relevance'``, R.
        """
        R = self.feature_relevance(X)
        r = R.sum(axis=1) + R.sum(axis=0)  # (n_latent, n_freqs)
        w = self.squaresum_3d.w.numpy()  # (n_freqs, n_nodes, n_latent)
        p = np.abs(np.einsum('ilf,fla->iaf', dcov, w))
        sd = p.std(axis=0)
        z = np.divide(p - p.mean(axis=0), sd, out=np.zeros_like(p),
                      where=sd > 0)
        total = r.sum()
        if total <= 0:
            raise ValueError("The features have no relevance (the model's "
                             "output doesn't depend on them)")
        return {'pattern': np.einsum('iaf,af->i', z, r) / total,
                'freq_relevance': r.sum(axis=0) / total,
                'filter_relevance': r / total,
                'relevance': R}

    def extract_combined_patterns(self, weight_paths=None):
        """Combined patterns (:meth:`combined_pattern`) of each CV fold's
        model, from its validation fold: the relevance and the input
        covariance both come from the whole fold.

        Parameters
        ----------
        weight_paths : list of str, optional
            The weights of each fold's model. Defaults to those saved by
            :meth:`train` with ``save_fold_weights=True``
            (:meth:`fold_weights_path`).

        Returns
        -------
        out : dict
            ``'pattern'`` (n_nodes, n_folds), ``'freq_relevance'``
            (n_freqs, n_folds), ``'filter_relevance'`` (n_latent,
            n_freqs, n_folds), ``'relevance'`` (n_latent, n_latent,
            n_freqs, n_folds; for ``extract_patterns(relevance=...)``),
            and ``'pointwise_standardized'`` (the standardized pointwise
            weights, folds last) and ``'pair_sd'`` (n_latent * n_latent,
            n_folds; for ``extract_patterns(standardized=...)``), all from
            each fold's model and validation data.
            The model is left with the last fold's
            weights.

        Raises
        ------
        FileNotFoundError
            If a fold's weights are missing (trained without
            ``save_fold_weights``).
        """
        n_folds = len(self.dataset.h_params['folds'][0])
        if weight_paths is None:
            weight_paths = [self.fold_weights_path(k)
                            for k in range(n_folds)]
        missing = [p for p in weight_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"No weights of {len(missing)} fold models, e.g. "
                f"{missing[0]}: train with save_fold_weights=True.")
        out = defaultdict(list)
        for fold, path in enumerate(weight_paths):
            self.km.load_weights(path)
            X = self._validation_data(fold)
            res = self.combined_pattern(X, input_covariance(X))
            activations = self._activations(X)
            res['pointwise_standardized'] = self._standardized_pointwise(
                activations)
            res['pair_sd'] = activations['pointwise'].reshape(
                len(activations['pointwise']), -1).std(axis=0)
            for key in ('pattern', 'freq_relevance', 'filter_relevance',
                        'relevance', 'pointwise_standardized', 'pair_sd'):
                out[key].append(res[key])
        return {key: np.stack(v, -1) for key, v in out.items()}
