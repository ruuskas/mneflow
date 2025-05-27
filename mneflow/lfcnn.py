# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:11:39 2025

@author: ipzub
"""

import tensorflow as tf

import numpy as np

from mne import channels, evoked, create_info, Info
from mne.filter import filter_data


from scipy.stats import spearmanr, pearsonr
from scipy.signal import welch

from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .layers import LFTConv, DeMixing, FullyConnected, TempPooling, LFTConvTranspose
from tensorflow.keras.layers import SeparableConv2D, Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.initializers import Constant
#from tensorflow.keras import regularizers as k_reg, constraints, layers

#from .layers import LSTM
#import csv
#import os
from .data import Dataset
#from .utils import regression_metrics, _onehot
from .models import BaseModel
from collections import defaultdict
from .losses import riemann_loss, CosMSE

class LFCNN(BaseModel):
    """LF-CNN. Includes basic parameter interpretation options.

    For details see [1].
    References
    ----------
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None, specs=None, specs_prefix=False):
        """

        Parameters
        ----------
        Dataset : mneflow.Dataset

        specs : dict
                dictionary of model hyperparameters {

        n_latent : int
            Number of latent components.
            Defaults to 32.

        nonlin : callable
            Activation function of the temporal convolution layer.
            Defaults to tf.nn.relu

        filter_length : int
            Length of spatio-temporal kernels in the temporal
            convolution layer. Defaults to 7.

        pooling : int
            Pooling factor of the max pooling layer. Defaults to 2

        pool_type : str {'avg', 'max'}
            Type of pooling operation. Defaults to 'max'.

        padding : str {'SAME', 'FULL', 'VALID'}
            Convolution padding. Defaults to 'SAME'.}

        stride : int
        Stride of the max pooling layer. Defaults to 2.

        """
        self.scope = 'lfcnn'
        if specs:
            meta.update(model_specs=specs)
        #specs = meta.model_specs
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        #specs.setdefault('model_path',  self.dataset.h_params['save_path'])
        super(LFCNN, self).__init__(meta, dataset, specs_prefix)
        #super().__init__(meta, dataset, specs_prefix)


    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """

        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
                            axis=3, specs=self.specs)
        self.dmx_out = self.dmx(self.inputs)

        self.tconv = LFTConv(size=self.specs['n_latent'],
                             nonlin=self.specs['nonlin'],
                             filter_length=self.specs['filter_length'],
                             padding=self.specs['padding'],
                             specs=self.specs
                             )
        self.tconv_out = self.tconv(self.dmx_out)

        self.pool = TempPooling(pooling=self.specs['pooling'],
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['stride'],
                                  padding='SAME'
                                  )
        self.pooled = self.pool(self.tconv_out)

        self.dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(self.pooled)

        # self.fin_fc0 = FullyConnected(size=self.specs['n_latent'], nonlin=tf.keras.activations.linear,
        #                     specs=self.specs)
        # fc0_out = self.fin_fc0(self.dropout)

        self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.keras.activations.linear,
                            specs=self.specs)
        #y_pred = self.fin_fc(fc0_out)
        y_pred = self.fin_fc(self.dropout)

        return y_pred

    def build_encoder(self):
        """Build computational graph for an interpretable Generator

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        encoder_specs = self.specs.copy()
        encoder_specs['l1_lambda'] = 0.
        self.km.trainable = False
        print("Freezing the decoder")

        # output_padding_height = max(0, min(strides[0]-1, diff_height))
        # output_padding_width = max(0, min(strides[1]-1, diff_width))
        # output_padding=(output_padding_height, output_padding_width)







        # # Calculate the difference
        # diff_height = target_shape[1] - default_output_height
        # diff_width = target_shape[2] - default_output_width


        #start with the output of decoder
        #elf.enc_inputs = layers.Input(self.y_shape)
        self.enc_fc = FullyConnected(scope='def', size=self.fin_fc.w.shape[0], nonlin=tf.identity,
                            specs=encoder_specs)
        enc_tconv_activations =  self.enc_fc(self.y_pred)
        print("enc_tconv_activations: ", enc_tconv_activations.shape, self.pooled.shape)
        self.enc_tconv_activations_r = tf.keras.layers.Reshape(self.pooled.shape[1:])


        enc_tconv_activations_r =  self.enc_tconv_activations_r(enc_tconv_activations)
        default_output_t = enc_tconv_activations_r.shape[2] * self.specs['stride']
        diff_padding = default_output_t - self.dataset.h_params['n_t']
        #print(diff_padding)
        if self.dataset.h_params['n_t']%self.specs['stride'] == 0:
            padding=None
        # elif self.dataset.h_params['n_t']%self.specs['stride'] == 1:
        #     padding = (1, 1)
            #padding = (self.dataset.h_params['n_t']%self.specs['stride'] - 1, 0)
        else :
            #padding = ((self.dataset.h_params['n_t'] - 1)%self.specs['stride'] , 0)
            padding = True

        n_pads = max(0, self.specs['stride'] - diff_padding)

        print("before upsampling: ", enc_tconv_activations_r.shape)
        enc_dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(enc_tconv_activations_r)
        #upool and apply transposed depthwise convolution
        print("pooled:", self.pooled.shape)
        deconv_type = 1
        if deconv_type == 1:
            if padding:
                padding = (n_pads, 1)
            # sep = tf.keras.layers.Conv2D(filters=self.specs['n_latent'],
            #                            kernel_size=1,
            #                            padding='same'
            #                            )(enc_dropout)
            # print('sep:', sep.shape)
            # Then use transposed convolution to reverse the spatial reduction
            self.enc_tconv_trans=tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=(self.specs['filter_length'], 2),
                strides=(self.specs['stride'], 1),
                padding='same',
                output_padding=padding,
                data_format='channels_first',
                dilation_rate=(1, 1),
                activation=None,
                use_bias=False,
                kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform',
                kernel_regularizer=None, #tf.keras.regularizers.l1(encoder_specs['l1_lambda']),
                bias_regularizer=None,
                activity_regularizer=None,
                #kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1]),
                bias_constraint=None,
            )
            enc_deconv0 = self.enc_tconv_trans(enc_dropout)
            print("deconv0: ", enc_deconv0.shape)
            enc_deconv1 = DeMixing(scope='dede',
                                   size=1,
                                   nonlin=tf.identity,
                                   axis=1, specs=encoder_specs)(enc_deconv0)

            enc_deconv = tf.keras.ops.transpose(enc_deconv1, [0, 3, 1, 2])
            # enc_deconv = tf.keras.layers.Conv2D(filters=1,
            #                                     kernel_size=(10, self.specs['n_latent']),
            #                                     strides=10,
            #                                     padding='same',
            #                                     data_format='channels_first',
            #                                     #dilation_rate=(1, 1),
            #                                     activation=None,
            #                                     use_bias=True,
            #                                     kernel_initializer='glorot_uniform',
            #                                     bias_initializer='glorot_uniform',
            #                                     )(enc_deconv0)
            print("deconv: ", enc_deconv.shape)
        elif deconv_type == 2:
            self.enc_tconv_trans=tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=(self.specs['filter_length'], self.specs['n_latent']),
                strides=(self.specs['stride'], 1),
                padding='same',
                output_padding=padding,
                data_format='channels_first',
                dilation_rate=(1, 1),
                activation='relu',
                use_bias=False,
                kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform',
                kernel_regularizer=None, #tf.keras.regularizers.l1(encoder_specs['l1_lambda']),
                bias_regularizer=None,
                activity_regularizer=None,
                #kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1]),
                bias_constraint=None,
            )

            enc_deconv = self.enc_tconv_trans(enc_dropout)
        #enc_deconv = tf.transpose(enc_deconv, perm=[0,3,1,2])

        print("Enc_deconv:", enc_deconv.shape)
        print("TCONV OUT:", self.tconv_out.shape)
        assert enc_deconv.shape == self.tconv_out.shape

        self.de_dmx = DeMixing(scope='dede',
                               size=self.dataset.h_params['n_ch'],
                               nonlin=tf.identity,
                               axis=3, specs=encoder_specs)

        self.X_pred = self.de_dmx(enc_deconv)
        print(self.X_pred.shape)

        self.km_enc = tf.keras.Model(inputs=self.inputs, outputs=self.X_pred)

        self.meta.train_params['enc_loss'] = [tf.keras.losses.CosineSimilarity(axis=[3]),
                                              #tf.keras.losses.MSE
                                              ]
        #self.meta.train_params['enc_loss'] = [CosMSE]

        #self.meta.train_params['enc_loss'] = [riemann_loss]
        #alpha = .5
        self.km_enc.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=self.meta.train_params['enc_loss'],
                        metrics=[tf.keras.metrics.RootMeanSquaredError(name="RMSE")],
                        #loss_weights=[alpha, 1.-alpha]
                        )

    def build_alt_encoder(self):
        """Build computational graph for an interpretable Generator

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        encoder_specs = self.specs.copy()
        encoder_specs['l1_lambda'] = 0.
        self.km.trainable = False
        print("Freezing the decoder")

        #enc_tconv_activations_r = self.tconv_out

        default_output_t = self.tconv_out.shape[2] * self.specs['stride']
        diff_padding = default_output_t - self.dataset.h_params['n_t']

        if self.dataset.h_params['n_t']%self.specs['stride'] == 0:
            padding=None
        else :
            padding = True
            n_pads = max(0, self.specs['stride'] - diff_padding)

        print("before upsampling: ", self.tconv_out.shape)

        self.enc_tconv_trans=tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(self.specs['filter_length'], self.specs['n_latent']),
            strides=(1, 1),
            padding='same',
            output_padding=padding,
            data_format='channels_first',
            dilation_rate=(1, 1),
            activation='relu',
            use_bias=False,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            kernel_regularizer=None, #tf.keras.regularizers.l1(encoder_specs['l1_lambda']),
            bias_regularizer=None,
            activity_regularizer=None,
            #kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1]),
            bias_constraint=None,
        )

        enc_deconv = self.enc_tconv_trans(self.tconv_out)

        print("Enc_deconv:", enc_deconv.shape)
        print("TCONV OUT:", self.tconv_out.shape)
        assert enc_deconv.shape == self.tconv_out.shape

        self.de_dmx = DeMixing(scope='dede',
                               size=self.dataset.h_params['n_ch'],
                               nonlin=tf.identity,
                               axis=3, specs=encoder_specs)

        self.X_pred = self.de_dmx(enc_deconv)
        print(self.X_pred.shape)

        self.km_enc = tf.keras.Model(inputs=self.inputs, outputs=self.X_pred)

        #self.meta.train_params['enc_loss'] = [tf.keras.losses.CosineSimilarity(axis=[2, 3]),
        #                                      tf.keras.losses.MAE]
        self.meta.train_params['enc_loss'] = [CosMSE]

        #self.meta.train_params['enc_loss'] = [riemann_loss]
        #alpha = .5
        self.km_enc.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=self.meta.train_params['enc_loss'],
                        metrics=[tf.keras.metrics.RootMeanSquaredError(name="RMSE")],
                        #loss_weights=[alpha, 1.-alpha]
                        )
    def compute_alt_enc_patterns(self, method='combined', use_y_cov=False):
        """

        """
        F, names = self.meta.get_feature_relevances(sorting=method,
                                               integrate=['timepoints'],
                                               diff=True)
        print("F:", F.shape)
        #n_components, n_y,
        n_folds = F.shape[-1]
        #W = self.weights['dmx'] #(n_ch, n_components, n_folds)

        if use_y_cov:
            y_cov = self.patterns['ccms']['cov_y_hat']

        class_topos = []
        fold_topos = []
        fold_ind = n_folds- 1
        class_topos = []

        #Computed weighted sum of spatial patterns weighted by their relevance
        a = tf.transpose(self.de_dmx.weights[0])
        print('a:', a.shape)

        class_topos.append(a)
        class_topos = np.stack(class_topos, 2) # (n_ch, n_components, n_folds)
        print('combined, class topos', class_topos.shape)
        if use_y_cov:
            w_out = np.dot(F[:, :, fold_ind],
                           np.linalg.inv(y_cov[:, :, fold_ind]))
            #print('F: ', F.shape)
            #print('w_out: ', w_out.shape)
            pattern = np.sum(class_topos * w_out[None, ...], axis=1)
            #print(pattern.shape)
        else:
            pattern = np.sum(class_topos * F[None, :, :, fold_ind], 1) #weighted sum over components
        fold_topos.append(pattern)
        patterns = np.stack(fold_topos, 2) #(n_ch, n_y, n_folds)
        print('combined:', patterns.shape)

            #patterns = self.aligned_mean(class_topos)

        return np.squeeze(patterns) #, unpooled_wfs

    def compute_enc_patterns(self, inputs=None):
        """

        """
        if not np.any(inputs):
            print('Using fake inputs')
            inputs = np.identity(self.out_dim)
        enc_fc_out = self.enc_fc(inputs)
        pooled = self.enc_tconv_activations_r(enc_fc_out)
        unpooled_wfs = self.enc_tconv_trans(pooled)
        patterns = self.de_dmx(unpooled_wfs)

        return np.squeeze(patterns) #, unpooled_wfs

    def train_encoder(self, n_epochs, eval_step=None, min_delta=1e-6,
              early_stopping=3, collect_patterns=False):
        n_folds = 1
        self.cv_enc_losses = []
        self.cv_enc_metrics = []
        dataset_train = self.dataset.train.map(lambda x, y : (x, x))
        dataset_val = self.dataset.val.map(lambda x, y : (x, x))
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=min_delta,
                                                      patience=early_stopping,
                                                      restore_best_weights=True)
        self.t_hist = self.km_enc.fit(dataset_train,
                                   validation_data=dataset_val,
                                   epochs=n_epochs, steps_per_epoch=eval_step,
                                   shuffle=True,
                                   validation_steps=self.dataset.validation_steps,
                                   callbacks=[stop_early], verbose=2,
                                   )
        losses, metrics = self.km_enc.evaluate(dataset_val,
                                           steps=self.dataset.validation_steps,
                                           verbose=0)
        self.cv_enc_losses.append(losses)
        self.cv_enc_metrics.append(metrics)

        print("""{} with {} fold(s) completed. \n
              Validation Performance:
              Loss: {:.4f}.
              Metric: {:.4f}"""
              .format("Encoder training with", n_folds,
                      np.mean(self.cv_enc_losses), np.mean(self.cv_enc_metrics)))



    def get_config(self):
            # Do not call super.get_config!
            # This gave an error for me.
            config = {
                "dmx": self.dmx,
                "dmx_out": self.dmx_out,
                "tocnv": self.tconv,
                "tconv_out": self.tconv_out,
                "pool": self.pool,
                "pooled": self.pooled,
                "dropout": self.dropout,
                "fin_fc": self.fin_fc
            }
            return config

    def _get_class_conditional_spatial_covariance(self, X, y):
        """Compute spatial class-conditional covariance matrix from the dataset

        Parameters:
        -----------
        dataset : tf.data.Dataset

        Returns : dcov [y_shape, n_ch, n_ch]

        """
        #TODO: Fix regression case
        dcovs = []
        dcovs_n = []
        for class_y in range(self.out_dim):
            class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)==class_y))#[0]
            xs = np.squeeze(X.numpy()[class_ind, ...])
            #xs -= np.mean(xs, axis=-2, keepdims=True)
            ddof_s = xs.shape[0]*self.dataset.h_params['n_t'] - 1
            cov_s = np.einsum('ijk, ijl -> kl', xs, xs) / ddof_s

            anti_class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)!=class_y))#[0]
            xn = np.squeeze(X.numpy()[anti_class_ind, ...])
            ddof_n = xn.shape[0]*self.dataset.h_params['n_t'] - 1
            cov_n = np.einsum('ijk, ijl -> kl', xn, xn) / ddof_n

            dcovs.append(cov_s) #  - cov_n
            dcovs_n.append(cov_n)
        return np.stack(dcovs, -1), np.stack(dcovs_n, -1)


    def patterns_cov_xx(self, y, weights, activations, dcov):
        """
        X - [i,...,m]
        y - [i,...,j] - used for cov[y]
        w - [k,...,j]
        Sx - [k,...,mj]
        """

        x_shape = list(activations['pooled'].shape)
        y_shape = list(y.shape)

        ddof = activations['pooled'].shape[0] - 1
        X = np.reshape(activations['pooled'], [activations['pooled'].shape[0], -1])

        y = np.reshape(y, [y.shape[0], -1])

        w = np.reshape(weights['out_weights'], [-1, weights['out_weights'].shape[-1]])
        assert(X.shape[-1] == w.shape[0]), 'Shape mismatch X:{} w:{}'.format(X.shape, w.shape)
        assert(y.shape[-1] == w.shape[1]), 'Shape mismatch y:{} w:{}'.format(y.shape, w.shape)
        X = X - X.mean(0, keepdims=True)
        cov_xx = np.einsum('ik ,ij -> kj ', X, X) / ddof

        #compute inverse covariance of the output
        cov_yy = np.einsum('ij, ik -> jk', y, y) / ddof
        prec_yy = tf.linalg.pinv(cov_yy)

        #compute directions of Sx: a = cov_xy*(cov_yy)**-1
        a_out = np.einsum('ii, ij, jj -> ij', cov_xx, w, prec_yy)

        Sx = np.einsum('il, il, ji -> jil', a_out, w, X)
        Sx = np.reshape(Sx, [-1, x_shape[-1], y_shape[-1]])
        Sx = Sx - Sx.mean(1, keepdims=True)
        ddof = Sx.shape[0] - 1
        cov_sx = np.einsum('ijk, ilk -> jlk', Sx, Sx) / ddof

        patterns = []

        for i in range(y.shape[-1]):
            prec_sx = np.linalg.pinv(cov_sx[...,i])
            dc = dcov['class_conditional'][..., i]
            patterns.append(np.einsum('hi, ij, jk -> h',
                                      dc, weights['dmx'], prec_sx))
        patterns = np.stack(patterns, -1)
        return patterns



    def patterns_cov_xy_hat(self, X, y, activations, weights):
        Sx_tconv = self.backprop_fc(activations['pooled'],
                                    activations['fc'],
                                    y,
                                    weights['out_weights'])
        Sx_dmx = self.backprop_covxy(X,
                                    activations['dmx'],
                                    Sx_tconv,
                                    weights['dmx'])
        return Sx_tconv, Sx_dmx


    def backprop_fc(self, X, y_hat, y, w):
        """
        X - [i,...,m]
        y - [i,...,j]
        w - [k,...,j]
        Sx - [k,...,mj]"""
        x_shape = list(X.shape)
        y_shape = list(y_hat.shape)

        ddof = X.shape[0] - 1
        X = np.reshape(X, [X.shape[0], -1])
        y = np.reshape(y, [y.shape[0], -1])
        y_hat = np.reshape(y_hat, [y_hat.shape[0], -1])
        w = np.reshape(w, [-1, w.shape[-1]])

        assert(X.shape[-1] == w.shape[0]), 'Shape mismatch X:{} w:{}'.format(X.shaep, w.shape)
        assert(y_hat.shape[-1] == w.shape[1]), 'Shape mismatch y:{} w:{}'.format(y_hat.shape, w.shape)
        X = X - X.mean(0, keepdims=True)
        y_hat = y_hat - y_hat.mean(0, keepdims=True)
        y = y - y.mean(0, keepdims=True)
        cov_xy = np.einsum('ik ,ij -> kj ', X, y_hat) / ddof

        cov_yy = np.einsum('ij, ik -> jk', y, y) / ddof
        prec_yy = tf.linalg.inv(cov_yy)

        #compute directions of Sx: a = cov_xy*(cov_yy)**-1
        a_out = np.einsum('jk, kl -> jl', cov_xy, prec_yy) #shape = [n_t_pooled, n_latent, n_classes]

        # A.*w
        #aw = a_out * w #shape = [...]
        # activation of tconv by each signal component of each sample
        #Sx = np.einsum('il, kl -> ik', y_hat, aw) #shape = [n_batch, ...]
        Sx = np.einsum('il, il, ji -> jil', a_out, w, X)
        Sx = np.squeeze(np.reshape(Sx, x_shape + y_shape[1:]))
        return Sx

    def backprop_covxy(self, X, Hx, Sx, w):
        xdmx = np.reshape(Hx, [-1, Hx.shape[-1]])
        xdmx = xdmx - xdmx.mean(0, keepdims=True)
        xinp = np.reshape(X, [-1, X.shape[-1]])
        xinp = xinp - xinp.mean(0, keepdims=True)
        cov_xy = np.dot(xinp.T, xdmx)
        print("aw", cov_xy.shape)
        aw = cov_xy
        sx = np.reshape(Sx, [-1, Sx.shape[-2], Sx.shape[-1]])
        sx = sx - sx.mean(0, keepdims=True)
        ddof = sx.shape[0] - 1
        cov_sx = np.einsum('ijk, ilk -> kjl', sx,sx) / ddof
        print("cov_sx:", cov_sx.shape)
        prec_yy_hat = np.stack([np.linalg.pinv(s) for s in cov_sx])
        print(w.shape, prec_yy_hat.shape)
        ww = np.einsum('ij, jlk -> ikl', w, prec_yy_hat)
        print(cov_xy.shape, ww.shape)
        a = np.einsum('ij, ijk -> ik', cov_xy, ww)

        return a

    def patterns_pinv_w(self, y, weights, activations, dcov):
        combined_topos = []
        pinv_dmx = np.linalg.pinv(weights['dmx']).T#np.dot(spatial_filters, np.linalg.inv(np.dot(spatial_filters.T, spatial_filters)))
        pinv_wfc = np.linalg.pinv(weights['out_w_flat']).T#np.dot(out_w_flat, np.linalg.inv(np.dot(out_w_flat.T, out_w_flat)))
        pinv_tck = np.linalg.pinv(weights['tconv']).T #np.dot(tconv_kernels, np.linalg.inv(np.dot(tconv_kernels.T, tconv_kernels)))

        #Least square singal estimate in tconv given wfc and fc_activations
        Sx_tconv = np.einsum('jk, ik ->ij', pinv_wfc, activations['fc'])
        Sx_tconv = np.reshape(Sx_tconv, activations['pooled'].shape)

        #Reverse pooling and depthwise convolution for each class
        Sx_dmx = []
        #dc = dcov['input_spatial']
        #n_padding = self.dataset.h_params['n_t']%self.specs['stride']
        for class_y in range(self.out_dim):
            class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)==class_y))#[0]
            Sxm = np.squeeze(Sx_tconv[class_ind, :].mean(0, keepdims=True))
            Sxm = np.atleast_2d(Sxm)
            dc = dcov['class_conditional'][..., class_y]
            combined_topos.append(np.einsum('hi,ij,tj->ht',
                                            dc,
                                            weights['dmx'],
                                            Sxm))
        topos = np.stack(combined_topos, 1)
        return topos


    def patterns_wfc_mean(self, y, weights, activations, dcov):
        combined_topos = []
        #uses y explicitely instead of cov[x,y]
        #accurate but has little to do with the model
        #dc = dcov['input_spatial']
        for class_y in range(self.out_dim):
            #compute mean activation of final layer for each class
            #TODO: -> to self.activations
            class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)==class_y))#[0]
            fc_bp_out = (np.dot(activations['fc'].numpy()[class_ind, :],
                               weights['out_w_flat'].T)).mean(0)

            fc_bp_out = fc_bp_out.reshape([activations['pooled'].shape[2],
                                           activations['pooled'].shape[3]],
                                           order='C')
            dc = dcov['class_conditional'][..., class_y]
            #fc_bp_out = np.maximum(fc_bp_out, 0)
            class_patterns = np.dot(dc,
                                    weights['dmx'])
            cp = np.einsum('ck, ik -> c', class_patterns, fc_bp_out)

            combined_topos.append(cp) # + spatial_biases[class_y]

        topos = np.stack(combined_topos, 1)
        return topos


    def compute_patterns(self, data_path=None, verbose=False):
        """Computes spatial patterns from filter weights.
        Required for visualization.

        Parameters
        ----------
        data_path : str or list of str
            Path to TFRecord files on which the patterns are estimated.

        output : str {'patterns, 'filters', 'full_patterns'}
            String specifying the output.

            'filters' - extracts weights of the spatial filters

            'patterns' - extracts activation patterns, obtained by
            left-multipying the spatial filter weights by the (spatial)
            data covariance.

            'full-patterns' - additionally multiplies activation
            patterns by the precision (inverse covariance) of the
            latent sources

        Returns
        -------
        self.patterns
            spatial filters or activation patterns, depending on the
            value of 'output' parameter.

        self.lat_tcs
            time courses of latent sourses.

        self.filters
            temporal convolutional filter coefficients.

        self.out_weights
            weights of the output layer.

        self.rfocs
            feature relevances for the output layer.
            (See self.get_output_correlations)

        Raises:
        -------
            AttributeError: If `data_path` is not specified.
        """
        from time import time
        self.nfft = 128
        patterns_struct = {'weights' : {'dmx':[], 'tconv':[], 'fc':[],
                                        'tconv_freq_resposes':{}},
                           'ccms' : {'dmx':[], 'tconv':[], 'fc':[], 'input':[], 'dmx_psd':[]},
                           'dcov' : {'input_spatial':[], 'class_conditional':[],
                                     'k-1':[]},
                           'patterns' : {},
                           'spectra': {},
                           'freqs': None
                           }

        if not data_path:
            print("Computing patterns: No path specified, using validation dataset (Default)")
            ds = self.dataset.val
        elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
            #TODO: rebalnce?
            ds = self.dataset._build_dataset(data_path,
                                             split=False,
                                             test_batch=None,
                                             repeat=True)
        elif isinstance(data_path, Dataset):
            if hasattr(data_path, 'test'):
                ds = data_path.test
            else:
                ds = data_path.val
        elif isinstance(data_path, tf.data.Dataset):
            ds = data_path
        else:
            raise AttributeError('Specify dataset or data path.')

        start = time()
        X, y = [row for row in ds.take(1)][0]
        ndof = X.shape[0] * self.dataset.h_params['n_t'] - 1

        #get layer activations
        activations = {}
        # Extract activations
        activations['dmx'] = self.dmx(X)
        activations['tconv'] = self.tconv(activations['dmx'])
        activations['pooled'] = self.pool(activations['tconv'])
        activations['fc']  = self.fin_fc(activations['pooled'])
        if verbose:
            print(""""Activations: \n
                  DMX: {}
                  TCONV: {}
                  POOLED: {}
                  FC_DENSE: {}""".format(
                  activations['dmx'].shape,
                  activations['tconv'].shape,
                  activations['pooled'],
                  activations['fc'].shape))

        stop = time() - start
        print("ACTIVATIONS:  {:.2f}s".format(stop))

        start = time()
        weights = self.extract_weights()
        spectra = self.compute_spectra(activations=activations)

        if not patterns_struct['freqs']:
            patterns_struct['freqs'] = spectra['freqs']

        stop = time() - start
        print("Weights and spectra: {:.2f}s".format(stop))
        #CCMs are mean activations of each layer for each class
        start = time()
        dcov = {}
        dcov['input_spatial'] = np.einsum('hijk, hijl -> kl', X, X) / ndof
        dcov['class_conditional'], dcov['k-1']  = self._get_class_conditional_spatial_covariance(X, y)
        stop = time() - start
        print("DCOVS:  {:.2f}s".format(stop))

        start = time()
        ##True evoked
        if self.dataset.h_params['target_type'] == 'float':
            self.true_evoked_data = X.numpy().mean(0)

            ccm_dmx = activations['dmx'].numpy().mean(0)[..., np.newaxis]

            ccm_tconv = activations['tconv'].numpy().mean(0)[..., np.newaxis]

            ccm_pooled = activations['pooled'].numpy().mean(0)[..., np.newaxis]

            ccm_fc = activations['fc'].numpy().mean(0)[..., np.newaxis]

            cov_y_hat = np.cov(tf.transpose(activations['fc'], perm=[1, 0]))
            cov_y = np.cov(tf.transpose(y, perm=[1, 0]))

        elif self.dataset.h_params['target_type'] == 'int':
            y_int = np.argmax(y, 1)
            y_unique = np.unique(y_int)
            evokeds = np.array([X.numpy()[y_int == i, ...].mean(0)
                                for i in y_unique])
            self.true_evoked_data = np.squeeze(evokeds)
            ccm_dmx = np.stack([activations['dmx'].numpy()[y_int == i, ...].mean(0)
                                for i in y_unique], -1)
            ccm_tconv = np.stack([activations['tconv'].numpy()[y_int == i, ...].mean(0)
                                for i in y_unique], -1)
            ccm_pooled = np.stack([activations['pooled'].numpy()[y_int == i, ...].mean(0)
                                for i in y_unique], -1)
            ccm_fc = np.stack([activations['fc'].numpy()[y_int == i, ...].mean(0)
                                for i in y_unique], -1)
            cov_y_hat = np.cov(tf.transpose(activations['fc'], perm=[1, 0]))
            cov_y = np.cov(tf.transpose(y, perm=[1, 0]))

        stop = time() - start
        print("CCMs:  {:.2f}s".format(stop))
        # compute the effect of removing each latent component on the cost function

        patterns_struct['weights'] = weights
        patterns_struct['spectra'] = spectra
        patterns_struct['dcov'] = dcov
        patterns_struct['ccms'] = {'dmx': ccm_dmx, # n_t, n_latent, n_classes
                                   'tconv':ccm_tconv, #n_t, n_latent, n_classes
                                   'pooled':ccm_pooled, #n_t_pooled, n_latent, n_classes
                                   'fc':ccm_fc, #n_y, n_y
                                   'cov_y_hat':cov_y_hat,
                                   'cov_y': cov_y,
                                   'psds': spectra['psds']} #n_y, n_y


        start = time()
        combined_patterns = self._compute_combined_patterns(y, weights, activations, dcov)
        stop = time() - start
        print("Combined patterns: {:.2}s".format(stop))
        patterns_struct.update(combined_patterns)

        #compute the effect of removing each latent component on the cost function
        start = time()
        patterns_struct['compwise_loss'] = self.compute_componentwise_loss(X, y, weights)
        stop = time() - start
        print("Compwise Loss: (Disabled) {:.2}s".format(stop))
        #correlation of fc activations to y
        start = time()
        patterns_struct['corr_to_output'] = self.get_output_correlations(activations, y)
        stop = time() - start
        print("Output corr: {:.2}s".format(stop))
        del X, activations

        return patterns_struct


    def _compute_combined_patterns(self, y, weights, activations, dcov):
        patterns = {'cov_xx':{}, 'pinv_w':{}, 'wfc_mean':{}}
        patterns['cov_xx']['spatial'] = self.patterns_cov_xx(y, weights, activations, dcov)

        patterns['pinv_w']['spatial'] = self.patterns_pinv_w(y, weights, activations, dcov).mean(-1)

        patterns['wfc_mean']['spatial'] = self.patterns_wfc_mean(y, weights, activations, dcov)
        return patterns

    def collect_patterns(self, fold=0, n_folds=1, n_comp=1,
                         methods=['weight',
                                  'compwise_loss',
                                  'weight_norm',
                                  'output_corr'
                                  ]):
        """
        Compute and store patterns during cross-validation.

        """

        patterns_struct = self.compute_patterns()

        # combined_methods = list(patterns_struct['patterns'].keys())
        # methods += combined_methods
        if len(self.cv_patterns.items()) == 0 or fold==0:
            n_ch = self.meta.data['n_ch']
            n_t_pooled = patterns_struct['weights']['out_weights'].shape[0]
            n_t = self.meta.data['n_t']
            n_fft = len(patterns_struct['freqs'])
            self.cv_patterns = defaultdict(dict)
            self.cv_patterns['freqs'] = patterns_struct['freqs']

            self.cv_patterns['dcov']['input_spatial'] = np.zeros([n_ch, n_ch,
                                                                  n_folds])
            self.cv_patterns['dcov']['class_conditional'] = np.zeros([n_ch, n_ch,
                                                                      self.y_shape[0],
                                                                      n_folds])
            self.cv_patterns['dcov']['k-1'] = np.zeros([n_ch, n_ch,
                                                        self.y_shape[0],
                                                        n_folds])
            self.cv_patterns['ccms']['dmx'] = np.zeros([n_t,
                                                        self.specs['n_latent'],
                                                        self.y_shape[0],
                                                        n_folds])

            self.cv_patterns['ccms']['tconv'] = np.zeros([n_t,
                                                        self.specs['n_latent'],
                                                        self.y_shape[0],
                                                        n_folds])

            self.cv_patterns['ccms']['pooled'] = np.zeros([self.pooled.shape[2],
                                                        self.specs['n_latent'],
                                                        self.y_shape[0],
                                                        n_folds])
            self.cv_patterns['ccms']['fc'] = np.zeros([self.y_shape[0],
                                                       self.y_shape[0],
                                                       n_folds])
            self.cv_patterns['ccms']['cov_y_hat'] = np.zeros([self.y_shape[0],
                                                              self.y_shape[0],
                                                              n_folds])
            self.cv_patterns['ccms']['cov_y'] = np.zeros([self.y_shape[0],
                                                              self.y_shape[0],
                                                              n_folds])
            self.cv_patterns['ccms']['cov_dmx'] = np.zeros([self.y_shape[0],
                                                              self.y_shape[0],
                                                              n_folds])
            self.cv_patterns['ccms']['cov_tconv'] = np.zeros([self.y_shape[0],
                                                              self.y_shape[0],
                                                              n_folds])

            self.cv_patterns['ccms']['psds'] = np.zeros([n_fft,
                                                             self.specs['n_latent'],
                                                              n_folds])



            for method in methods:
                self.cv_patterns[method]['feature_relevance'] = np.zeros([n_t_pooled,
                                                             self.specs['n_latent'],
                                                             self.y_shape[0],
                                                             n_folds])


        self.cv_patterns['dcov']['input_spatial'][:, :, fold] = patterns_struct['dcov']['input_spatial']
        self.cv_patterns['dcov']['class_conditional'][:, :, :, fold] = patterns_struct['dcov']['class_conditional']

        self.cv_patterns['compwise_loss']['feature_relevance'][:, :, :, fold] = patterns_struct['compwise_loss']
        self.cv_patterns['output_corr']['feature_relevance'][:, :, :, fold] = patterns_struct['corr_to_output']
        self.cv_patterns['weight']['feature_relevance'][:, :, :, fold] = patterns_struct['weights']['out_weights']
        [self.cv_weights[k].append(patterns_struct['weights'][k])
         for k in patterns_struct['weights'].keys()]

        self.cv_patterns['ccms']['dmx'][:, :, :, fold] = patterns_struct['ccms']['dmx'] #n_t, n_latent, n_classes, n_folds
        self.cv_patterns['ccms']['tconv'][:, :, :, fold] = patterns_struct['ccms']['tconv'] #n_t, n_latent, n_classes, n_folds
        self.cv_patterns['ccms']['pooled'][:, :, :, fold] = patterns_struct['ccms']['pooled'] #n_pooled, n_latent, n_classes, n_folds

        self.cv_patterns['ccms']['fc'][:, :, fold] = patterns_struct['ccms']['fc'] # n_logits, n_classes, n_folds
        self.cv_patterns['ccms']['cov_y_hat'][:, :, fold] = patterns_struct['ccms']['cov_y_hat'] # n_classes, n_classes, n_folds
        self.cv_patterns['ccms']['cov_y'][:, :, fold] = patterns_struct['ccms']['cov_y'] # n_classes, n_classes, n_folds
        self.cv_patterns['ccms']['psds'][:, :, fold] = patterns_struct['ccms']['psds']

    def compute_spectra(self, activations, nfft=128):
        ##Psds
        """
        Returns PSDs of latent components after spatial filtering.
        (n_fft, n_comonents)
        """
        psds = []
        for i in range(self.specs['n_latent']):

            ltc = activations['dmx'][:, 0, :, i] - np.mean(activations['dmx'][:, 0, :, i], axis=1, keepdims=True)
            fr, psd = welch(ltc,
                            fs=self.dataset.h_params['fs'],
                            nfft=nfft * 2,
                            nperseg=nfft)
            if len(fr[:-1]) < nfft:
                nfft = len(fr[:-1])
            psds.append(psd[:, 1:].mean(0))


        spectra = {}
        spectra['psds'] = np.array(psds).T
        spectra['freqs'] = fr[1:]
        spectra['nfft'] = nfft
        print(spectra['psds'].shape)
        return spectra



    def extract_weights(self, verbose=False):
        weights = {}

        # Extract weights

        # Spatial extraction fiters
        weights['dmx'] = np.squeeze(self.dmx.w.numpy())
        weights['dmx_b'] = self.dmx.b_in.numpy()
        # Temporal kernels
        weights['tconv'] = np.squeeze(self.tconv.filters.numpy())
        weights['tconv_b'] = np.squeeze(self.tconv.b.numpy())
        # Final layer
        weights['out_w_flat'] = self.fin_fc.w.numpy()
        weights['out_weights'] = np.reshape(self.fin_fc.w.numpy(),
                                 [self.pooled.shape[2],
                                  self.dmx.size,
                                  self.out_dim],
                                 order='C')

        weights['fc_b'] = self.fin_fc.b.numpy()

        if verbose:
            print(""""Weights: \n
                  DMX: {}
                  TCONV: {}
                  FC_DENSE: {}""".format(weights['dmx'].shape,
                  weights['tconv'].shape,
                  weights['out_weights'].shape))

        return weights

    def compute_componentwise_loss(self, X, y, weights):

        """
        Compute component relevances by recursive elimination
        """
        model_weights = self.km.get_weights()
        original_weights = model_weights.copy()
        base_loss, base_performance = self.km.evaluate(X, y, verbose=0)

        feature_relevances_loss = []
        n_out_t = weights['out_weights'].shape[0]
        n_out_y = weights['out_weights'].shape[-1]

        losses = np.zeros([self.specs['n_latent'], n_out_y])

        for jj in range(n_out_y):
            #for each class
            for i in range(self.specs["n_latent"]):

                new_weights = weights['out_weights'].copy()
                new_bias = weights['fc_b'].copy()
                new_weights[:, i, jj] = 0
                new_bias[jj] = 0
                new_weights = np.reshape(new_weights, weights['out_w_flat'].shape)
                model_weights[-2] = new_weights
                model_weights[-1] = new_bias
                self.km.set_weights(model_weights)
                new_loss = self.km.evaluate(X, y, verbose=0)[0]

                #loss_per_component.append(base_loss - loss)
                losses[i, jj] = new_loss - base_loss # larger difference is better
        losses = np.repeat(losses[np.newaxis, ...], n_out_t, axis=0)
        print(losses.shape)
        self.km.set_weights(original_weights)
        return losses


    def get_output_correlations(self, activations, y_true):
        """Computes a similarity metric between each of the extracted
        features and the target variable.

        The metric is a Manhattan distance for dicrete targets, and
        Spearman correlation for continuous targets.
        """
        corr_to_output = []
        y_true = y_true.numpy()
        flat_feats = activations['pooled'].numpy().reshape(y_true.shape[0], -1)


        for y_ in y_true.T:
            if self.dataset.h_params['target_type'] in ['float', 'signal']:
                rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])
                corr_to_output.append(rfocs.reshape(activations['pooled'].shape[1:]))


            elif self.dataset.h_params['target_type'] == 'int':
                rfocs = np.array([pearsonr(y_, f)[0] for f in flat_feats.T])

                corr_to_output.append(rfocs.reshape(activations['pooled'].shape[1:]))


        corr_to_output = np.concatenate(corr_to_output, 0).transpose([1, 2, 0])
        #print(corr_to_output.shape)
        if np.any(np.isnan(corr_to_output)):
            corr_to_output[np.isnan(corr_to_output)] = 0
        return corr_to_output

    # --- LFCNN plot functions ---

    def plot_evoked_peaks(self, data=None, t=None, class_subset=None,
                          sensor_layout='Vectorview-mag', title=None, savefig=None):
        """
        Plot one spatial topography of class-conditional average of the input.
        If timepoint is not specified it is picked as a maximum RMS for each
        class.


        Parameters
        ----------
        topos : np.array
            [n_ch, n_t, n_classes]
        sensor_layout : TYPE, optional
            DESCRIPTION. The default is 'Vectorview-mag'.

        """
        n = self.out_dim

        if data is None:
            data = self.true_evoked_data
            if not title:
                title = 'True Patterns'
        else:
            title = 'Model-derived patterns'

        if t is None:
            t = np.argmax(np.mean(data**2, axis=0).mean(-1))
            print(t)
        title = title +  't={}'.format(t)
        ed = np.stack([data[i, t, :] for i in range(n)], axis=-1)
        assert ed.ndim==2
        topoplot = self.plot_topos(ed, sensor_layout=sensor_layout,
                                   class_subset=class_subset, title=title)

        #topoplot.figure.suptitle(title)
        topoplot.show()
        if savefig:
            figname = '-'.join([self.meta.data['path'] + self.scope, self.meta.data['data_id'], title, "topos.svg"])
            topoplot.savefig(figname, format='svg', transparent=True)
        return topoplot

    def plot_topos(self, topos, sensor_layout='Vectorview-mag', class_subset=None,
                   title="Class %g"):
        """
        Plot any spatial distribution in the sensor space.
        TODO: Interpolation??


        Parameters
        ----------
        topos : np.array
            [n_ch, n_classes, ...]
        sensor_layout : TYPE, optional
            DESCRIPTION. The default is 'Vectorview-mag'.
        class_subset  : np.array, optional

        Returns
        -------
        None.

        """

        if topos.ndim > 2:
            topos = topos.mean(-1)
        topos_new = topos / topos.std(0, keepdims=True)

        n = topos.shape[1]

        if class_subset is None:
            class_subset = np.arange(0,  n, 1.)

        fake_evoked = self.make_fake_evoked(topos_new, sensor_layout)

        ft = fake_evoked.plot_topomap(times=class_subset,
                                    colorbar=True,
                                    scalings=1,
                                    time_format=title,
                                    outlines='head',
                                    #vlim= np.percentile(topos, [5, 95])
                                    )
        #ft.show()
        return ft

    def make_fake_evoked(self, topos, sensor_layout):
        if 'info' not in self.meta.data.keys():
            lo = channels.read_layout(sensor_layout)
            #lo = channels.generate_2d_layout(lo.pos)
            info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
            orig_xy = np.mean(lo.pos[:, :2], 0)
            for i, ch in enumerate(lo.names):
                if info['chs'][i]['ch_name'] == ch:
                    info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/4.5
                    #info['chs'][i]['loc'][4:] = 0
                else:
                    print("Channel name mismatch. info: {} vs lo: {}".format(
                        info['chs'][i]['ch_name'], ch))
        #info['sfreq'] = 1
        fake_evoked = evoked.EvokedArray(topos, info)
        return fake_evoked


    def explore_components(self, patterns_struct, sorting='output_corr',
                         integrate='max', info=None, sensor_layout='Vectorview-grad',
                         class_names=None):
        """Plots the weights of the output layer.

        Parameters
        ----------

        pat : int [0, self.specs['n_latent'])
            Index of the latent component to higlight

        t : int [0, self.h_params['n_t'])
            Index of timepoint to highlight

        Returns
        -------
        figure :
            Imshow [n_latent, y_shape]

        """
        self.meta.explore_components()




    def plot_waveforms(self, patterns_struct, sorting='weight', tmin=0, class_names=None,
                       bp_filter=False, tlim=None, apply_kernels=False):
        """Plots timecourses of latent components.

        Parameters
        ----------
        tmin : float
            Beginning of the MEG epoch with regard to reference event.
            Defaults to 0.


        sorting : str
            heuristic for selecting relevant components. See LFCNN._sorting
        """

        order, _ = self._sorting(patterns_struct, sorting)
        self.uorder = order.ravel()
        waveforms = patterns_struct['ccms']['tconv']

            #self.uorder = np.squeeze(order)
        #print(self.uorder)
        if not class_names:
            class_names = ["Class {}".format(i) for i in range(self.y_shape[-1])]

        f, ax = plt.subplots(2, 2)
        f.set_size_inches([16, 16])
        if np.any(self.uorder):
            #for jj, uo in enumerate(self.uorder):
            nt = self.dataset.h_params['n_t']

            tstep = 1/float(self.dataset.h_params['fs'])
            times = tmin + tstep*np.arange(nt)
            if apply_kernels:
                scaled_waveforms = np.array([np.convolve(kern, wf, 'same')
                            for kern, wf in zip(self.filters, self.waveforms)])
                #scaled_waveforms =(scaled_waveforms - scaled_waveforms.mean(-1, keepdims=True))  / (2*scaled_waveforms.std(-1, keepdims=True))
            else:
                #scaling = 3*np.mean(np.std(self.waveforms, -1))

                scaled_waveforms = (waveforms - waveforms.mean(-1, keepdims=True))  / (2*waveforms.std(-1, keepdims=True))
            if bp_filter:
                scaled_waveforms = scaled_waveforms.astype(np.float64)
                scaled_waveforms = filter_data(scaled_waveforms,
                                                  self.dataset.h_params['fs'],
                                                  l_freq=bp_filter[0],
                                                  h_freq=bp_filter[1],
                                                  method='iir',
                                                  verbose=False)
            [ax[0, 0].plot(times, wf, color='tab:grey', alpha=.25)
             for i, wf in enumerate(scaled_waveforms) if i not in self.uorder]

            [ax[0, 0].plot(times,
                          scaled_waveforms[uo],
                          linewidth=2., label=class_names[i], alpha=.75)
             for i, uo in enumerate(self.uorder)]
            ax[0, 0].set_title('Latent component waveforms')
            if tlim:
                ax[0, 0].set_xlim(tlim)

            tstep = float(self.specs['stride'])/self.dataset.h_params['fs']
            strides1 = np.arange(times[0], times[-1] + tstep/2, tstep)
            ax[1, 0].pcolor(strides1, np.arange(self.specs['n_latent']),
                           np.mean(self.tc_out, 0).T, #shading='auto'
                           )

            ax[1, 0].set_title("Avg. Temporal Convolution Output")
            ax[1, 0].set_ylabel("Component index")
            ax[1, 0].set_xlabel("Time, s")
            if tlim:
                ax[1, 0].set_xlim(tlim)
            if not hasattr(self, 'pattern_weights'):
                pattern_weights = np.einsum('ijk, jkl ->ikl', self.tc_out, self.out_weights)
                self.pattern_weights = np.maximum(pattern_weights + self.out_biases[None, :], 0.).mean(0)

            a = ax[0, 1].pcolor(self.pattern_weights, cmap='bone_r')
            divider = make_axes_locatable(ax[0,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(a, cax=cax, orientation='vertical')
            r = [ptch.Rectangle((i, uo), width=1,
                                height=1, angle=0.0) for i, uo in enumerate(self.uorder)]
            pc = collections.PatchCollection(r, facecolor=None, alpha=.5,
                                             linewidth=2.,
                                             edgecolor='tab:orange')
            ax[0, 1].add_collection(pc)

            ax[0, 1].set_title("Pattern weights")
            ax[0, 1].set_ylabel("Component index")
            ax[0, 1].set_xticks(np.arange(0.5, 0.5+len(class_names), 1))
            ax[0, 1].set_xticklabels(class_names)
            rpss = []
            for i, flt in enumerate(self.filters.T):

                flt -= flt.mean()
                h = self.freq_responses[i, :]
                psd = self.psds[i, :]

                #rpss.append(h/np.sum(h))
                rpss.append((psd*h)) #%)/np.sum(psd*h)

            [ax[1, 1].plot(self.freqs, rpss[uo], linewidth=2.5, label=class_names[i])
                             for i, uo in enumerate(self.uorder)]
            ax[1, 1].set_xlim(0, 90.)
            ax[1, 1].set_title("Relative power, %")
            ax[1, 1].set_xlabel("Frequency, Hz")
            ax[1, 1].legend()
            plt.show()
            return




    # def single_component_pattern(self, patterns_struct, sorting='weight',
    #                              n_comp=1):
    #     """Pick single component according to sorting method.
    #     Multiply spatial filter by data covariance"""
    #     order, ts = self._sorting( patterns_struct, sorting, n_comp=n_comp)
    #     c_topos = []
    #     c_psds = []
    #     c_frs = []
    #     #print(sorting, order)
    #     w = patterns_struct['weights']['dmx']
    #     #a = np.dot(patterns_struct['dcov']['input_spatial'], w)
    #     #print(sorting, ': component #{}', order[0][0])
    #     #a = np.dot(patterns_struct['dcov']['input_spatial'], w)
    #     for i, comps in enumerate(order):
    #         a = np.dot(patterns_struct['dcov']['class_conditional'][..., i], w)
    #         c_topos.append(a[:, comps])
    #         c_psds.append(patterns_struct['spectra']['psds'][comps, :].T,)
    #         #c_frs.append(patterns_struct['spectra']['freq_responses'][comps, :].T)
    #     topo = np.concatenate(c_topos, axis=-1)
    #     #freq_response = np.concatenate(c_frs, axis=-1)
    #     psd = np.concatenate(c_psds, axis=-1)
    #     #freqs = patterns_struct['freqs']
    #     return topo, psd#, freqs

    def plot_combined_pattern(self, method='weight', sensor_layout=None,
                              names=None, n_comp=1, plot_true_evoked=False,
                              savefig=None):
        if not names:
            names = ['Class {}'.format(i) for i in range(self.y_shape[-1])]


#        cc = np.array([np.corrcoef(self.cv_patterns[:, i, :].T)[i,:]
#               for i in range(self.cv_patterns.shape[1])])
        if len(self.cv_patterns.items()) > 0:
            print("Restoring from:", method )
            topos = np.mean(self.cv_patterns[method]['spatial'],
                                     -1)
            filters = np.mean(self.cv_patterns[method]['temporal'],
                                       -1)
            psds = np.mean(self.cv_patterns[method]['psds'],
                                    -1)
            #freqs = self.freqs


        elif method in ['weight', 'compwise_loss']:
            topos, filters, psds = self.single_pattern(sorting=method,
                                                       n_comp=n_comp)


        freqs = self.cv_patterns['freqs']

        topos /= np.maximum(topos.std(axis=0, keepdims=True),
                                     1e-3)
        n = self.y_shape[0]
        ncols = n
        lo = channels.read_layout(sensor_layout)
        #lo = channels.generate_2d_layout(lo.pos)
        info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
        orig_xy = np.mean(lo.pos[:, :2], 0)
        for i, ch in enumerate(lo.names):
            if info['chs'][i]['ch_name'] == ch:
                info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/4.5
                #info['chs'][i]['loc'][4:] = 0
            else:
                print("Channel name mismatch. info: {} vs lo: {}".format(
                    info['chs'][i]['ch_name'], ch))

        self.fake_evoked = evoked.EvokedArray(topos, info)
        self.fake_evoked.data[:, :n] = topos

        fake_times = np.arange(0,  n, 1.)
        ft = self.fake_evoked.plot_topomap(times=fake_times,
                                          #axes=ax[0, 0],
                                          colorbar=True,
                                          #vmax=vmax,
                                          scalings=1,
                                          time_format=method,
                                          #title='',
                                          #size=1,
                                          outlines='head',
                                          )
        #method = "paternnet_rect_cc_covdif_cc_fcactdif"
        #ft.set_size_inches([15, 3.5])
        if savefig:
            figname = '-'.join([self.meta.data['path'] + method, "topos.svg"])
            ft.savefig(figname, format='svg', transparent=True)
        if plot_true_evoked:
            #true_times = np.argmax(np.mean(self.true_evoked_data**2, -1),1)
            #ed = np.stack([self.true_evoked_data[i, tt, :] for i, tt in enumerate(true_times)])
            t = self.plot_evoked_peaks(None, sensor_layout=sensor_layout,
                                       title='True evoked')
            figname = '-'.join([self.meta.data['path'] + self.scope, self.meta.data['data_id'], 'true', "topos.svg"])
            t.savefig(figname, format='svg', transparent=True)