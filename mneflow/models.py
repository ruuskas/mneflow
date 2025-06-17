# -*- coding: utf-8 -*-
"""
Define mneflow.models.Model parent class and the implemented models as
its subclasses. Implemented models inherit basic methods from the
parent class.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""

#TODO: update vizualizations

import tensorflow as tf

import numpy as np

from typing import Callable
from mne import channels, evoked, create_info, Info
from mne.filter import filter_data


from scipy.stats import spearmanr, pearsonr
from scipy.signal import welch

from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .layers import LFTConv, VARConv, DeMixing, FullyConnected, TempPooling
from tensorflow.keras.layers import SeparableConv2D, Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras import regularizers as k_reg, constraints, layers

from .layers import LSTM
import csv
import os
from .data import Dataset
from .utils import regression_metrics, _onehot
from collections import defaultdict


def uniquify(seq):
    un = []
    [un.append(i) for i in seq if not un.count(i)]
    return un


# ----- Base model -----
#@tf.keras.utils.register_keras_serializable(package="mneflow")
class BaseModel():
    """Parent class for all MNEflow models.

    Provides fast and memory-efficient data handling and simplified API.
    Custom models can be built by overriding _build_graph and
    _set_optimizer methods.
    """

    def __init__(self, meta=None, dataset=None, specs_prefix=False):
        """
        Parameters
        ----------
        Dataset : mneflow.Dataset
            `Dataset` object.

        specs : dict
            Dictionary of model-specific hyperparameters. Must include
            at least `model_path` - path for saving a trained model
            See `Model` subclass definitions for details. Unless otherwise
            specified uses default hyperparameters for each implemented model.
        """
        self.specs = meta.model_specs
        meta.model_specs['model_path'] = os.path.join(meta.data['path'],
                                                      'models')

        self.meta = meta
        self.model_path = meta.model_specs['model_path'] #os.path.join(meta.data['path'], 'models')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


        if dataset:
            self.dataset = dataset
        elif not dataset and meta:
            self.dataset = Dataset(meta, **meta.data)
        else:
            print("Provide Dataset or Metadata file")

        self.input_shape = (self.dataset.h_params['n_seq'],
                            self.dataset.h_params['n_t'],
                            self.dataset.h_params['n_ch'])
        self.y_shape = self.dataset.y_shape
        self.out_dim = np.prod(self.y_shape)
        self.inputs = layers.Input(shape=(self.input_shape))
        #self.trained = False
        self.y_pred = self.build_graph()
        self.log = dict()
        self.cm = np.zeros([self.y_shape[-1], self.y_shape[-1]])
        self.cv_patterns = defaultdict(dict)
        self.cv_weights = defaultdict(list)
        if not hasattr(self, 'scope'):
            self.scope = 'basemodel'

        if specs_prefix:
           self.specs_prefix = '_'.join([str(v).replace('.', '-') for k,v in self.specs.items() if k not in ['nonlin', 'model_path', 'l1_scope', 'l2_scope', 'unitnorm_scope', 'scope']])
        else:
            self.specs_prefix = ''
        self.model_name = "_".join([self.scope,
                                    meta.data['data_id']])



    def build(self, optimizer="adam",
              loss=None,
              metrics=None, mapping=None,
              learn_rate=3e-4):
        """Compile a model.

        Parameters
        ----------
        optimizer : str, tf.optimizers.Optimizer
            Deafults to "adam"

        loss : str, tf.keras.losses.Loss
            Defaults to MSE in target_type is "float" and
            "softmax_crossentropy" if "target_type" is int

        metrics : str, list of str, tf.keras.metrics.Metric
            Defaults to RMSE in target_type is "float" and
                "categorical_accuracy" if "target_type" is int

        learn_rate : float
            Learning rate, defaults to 3e-4

        mapping : str

        """
        # Initialize computational graph
        if mapping:
            map_fun = tf.keras.activations.get(mapping)
            self.y_pred = map_fun(self.y_pred)

        self.km = tf.keras.Model(inputs=self.inputs, outputs=self.y_pred)

        params = {"optimizer": tf.optimizers.get(optimizer).from_config(
                                            {"learning_rate":learn_rate})}

        if loss:
            params["loss"] = tf.keras.losses.get(loss)
            loss_name = loss

        if metrics:
            if not isinstance(metrics, list):
                metrics = [metrics]
            params["metrics"] = [tf.keras.metrics.get(metric) for metric in metrics]

       # Initialize optimizer
        if self.dataset.h_params["target_type"] in ['float', 'signal']:
            params.setdefault("loss", tf.keras.losses.MeanSquaredError(name='MSE'))

            params.setdefault("metrics", [tf.keras.metrics.RootMeanSquaredError(name="RMSE")])

        elif self.dataset.h_params["target_type"] in ['int']:
            params.setdefault("loss", tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                                   name='Cat_CE'))
            params.setdefault("metrics", [tf.keras.metrics.CategoricalAccuracy(name="Cat_Acc")])

        self.km.compile(optimizer=params["optimizer"],
                        loss=params["loss"],
                        metrics=params["metrics"])

        if not loss and self.dataset.h_params["target_type"] in ['float', 'signal']:
            loss_name = 'MSE'
        elif not loss:
            loss_name = 'Cat_CE'
        else:
            loss_name = params['loss'].name
        _ = params.pop('loss')
        metrics = params.pop('metrics')
        metric_names = ':'.join([m.name for m in metrics])

        param_names = {k: v.name for k,v in params.items()}
        param_names['metrics'] = metric_names
        param_names['loss'] = loss_name
        param_names['learn_rate'] = learn_rate
        param_names['trained'] = False
        self.meta.update(train_params=param_names)

        self.km.save_weights(os.path.join(self.model_path,
                                          ''.join([self.model_name,
                                                   self.specs_prefix,
                                                   '_init.weights.h5'])))


        print('Input shape:', self.input_shape)
        print('y_pred:', self.y_pred.shape)
        print('Initialization complete!')

    def build_graph(self):
        """Build computational graph using defined placeholder self.X
        as input.

        Can be overriden in a sub-class for customized architecture.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """

        flat = Flatten()(self.inputs)
        self.fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                        specs=self.specs)
        y_pred = self.fc(flat)
        return y_pred


    def train(self, n_epochs=10, eval_step=None, min_delta=1e-6,
              early_stopping=3, mode='single_fold', prune_weights=False,
              collect_patterns=False, class_weights=None,
              noisy_labels=False, noise_std=.1) :

        """
        Train a model

        Parameters
        -----------

        n_epochs : int
            Maximum number of training eopchs.

        eval_step : int, None
            iterations per epoch. If None each epoch passes the training set
            exactly once

        early_stopping : int
            Patience parameter for early stopping. Specifies the number
            of epochs's during which validation cost is allowed to
            rise before training stops.

        min_delta : float, optional
            Convergence threshold for validation cost during training.
            Defaults to 1e-6.

        mode : str, optional
            can be 'single_fold', 'cv', 'loso'. Defaults to 'single_fold'

        collect_patterns : bool
            Whether to compute and store patterns after training each fold.

        class_weights : None, dict
            Whether to apply cutom wegihts fro each class

        noisy_labels : bool, optional
            Train model with addition gaussinan noise to labels. (Experimental)
            Does not work with class_weights

        noise_std : float, optional
            Standard deviation of the noise added to labels. (Experimental)
            Does not work with class_weights
        """


        if not eval_step:
            train_size = self.dataset.h_params['train_size']
            eval_step = train_size // self.dataset.h_params['train_batch'] + 1

        train_params = dict(n_epochs=n_epochs,
                            eval_step=eval_step,
                            early_stopping=early_stopping,
                            mode=mode,
                            min_delta=min_delta)

        self.meta.update(train_params=train_params)


        rmss = defaultdict(list)

        self.cv_losses = []
        self.cv_metrics = []
        self.cv_test_losses = []
        self.cv_test_metrics = []
        self.cv_metric_pvalues = []

        if class_weights:
            multiplier = 1. / min(class_weights.values())
            class_weights = {k:v*multiplier for k,v in class_weights.items()}

        else:
            class_weights = None
        print("Class weights: ", class_weights)

        if mode == 'single_fold':
            n_folds = 1
        elif mode == 'cv':
            n_folds = len(self.dataset.h_params['folds'][0])
            print("Running cross-validation with {} folds".format(n_folds))
        elif mode == "loso":
            n_folds = len(self.dataset.h_params['train_paths'])


        for jj in range(n_folds):

            print("Running {} fold: {}".format(mode, jj))
            if mode == "loso":
                test_subj = self.dataset.h_params['train_paths'][jj]
                train_subjs = self.dataset.h_params['train_paths'].copy()
                train_subjs.pop(jj)

                train, val = self.dataset._build_dataset(train_subjs,
                                                   train_batch=self.dataset.training_batch,
                                                   test_batch=self.dataset.validation_batch,
                                                   split=True, val_fold_ind=0)

            else:

                train, val = self.dataset._build_dataset(self.dataset.h_params['train_paths'],
                                                   train_batch=self.dataset.training_batch,
                                                   test_batch=self.dataset.validation_batch,
                                                   split=True, val_fold_ind=jj)
            if not noisy_labels:
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=self.meta.train_params['min_delta'],
                                                              patience=self.meta.train_params['early_stopping'],
                                                              restore_best_weights=True)
                stop_early.best = np.inf
                self.t_hist = self.km.fit(train,
                                   validation_data=val,
                                   epochs=self.meta.train_params['n_epochs'],
                                   steps_per_epoch=self.meta.train_params['eval_step'],
                                   shuffle=True,
                                   validation_steps=self.dataset.validation_steps,
                                   callbacks=[stop_early], verbose=2,
                                   class_weight=class_weights)
            else:
                model_path = os.path.join(self.model_path,
                                          ''.join([self.model_name,
                                                   self.specs_prefix]))
                trainer = NoisyTrainer(self.km,
                                       model_path = model_path,
                                       noise_std = noise_std,  # Noise standard deviation
                                       patience = self.meta.train_params['early_stopping'],     # Stop if no improvement for 5 epochs
                                       min_delta = self.meta.train_params['min_delta'] # Minimum change to count as improvement
                                       )

                self.t_hist = trainer.train(train, val,
                                            epochs=self.meta.train_params['n_epochs'],
                                            eval_step=self.meta.train_params['eval_step']
                                            )

            self.meta.train_params.update({"trained":True})

            v_loss, v_metric = self.evaluate(val)
            self.cv_losses.append(v_loss)
            self.cv_metrics.append(v_metric)

            if mode == 'loso':
                print("Creating loso test DS")
                test = self.dataset._build_dataset(test_subj,
                                                   test_batch=None,
                                                   split=False,
                                                   repeat=False)
            elif len(self.dataset.h_params['test_paths']):
                test = self.dataset._build_dataset(self.dataset.h_params['test_paths'],
                                                   test_batch=None,
                                                   split=False,
                                                   repeat=False)
            else:
                test = None

            if test:

                t_loss, t_metric = self.evaluate(test)
                self.cv_test_losses.append(t_loss)
                self.cv_test_metrics.append(t_metric)



            y_true, y_pred = self.predict(val)


            if self.dataset.h_params['target_type'] == 'float':
                rms = regression_metrics(y_true, y_pred)
                for k,v in rms.items():
                    rmss[k].append(v)
                print("Validation set: Corr =", rms['cc'], " R2 =", rms['r2'])

            else:
                self.cm += self._confusion_matrix(y_true, y_pred)
                rms = None

            if collect_patterns and self.scope == 'lfcnn':
                self.collect_patterns(fold=jj, n_folds=n_folds,
                                      n_comp=int(collect_patterns))


            if jj < n_folds - 1:
                self.km.load_weights(os.path.join(self.model_path,
                                                  ''.join([self.model_name,
                                                           self.specs_prefix,
                                                           '_init.weights.h5'])),
                                     skip_mismatch=True)
                self.shuffle_weights()


            else:
                print("Not shuffling the weights for the last fold")

            print("""Fold: {} Validation performance:\n
                  Loss: {:.4f},
                  Metric: {:.4f}""".format(jj, v_loss, v_metric))


        metrics = self.cv_metrics
        losses = self.cv_losses

        print("""{} with {} folds completed.
              Loss: {:.4f} +/- {:.4f}.
              Metric: {:.4f} +/- {:.4f}""".format(mode, n_folds,
                                                  np.mean(losses), np.std(losses),
                                                  np.mean(metrics), np.std(metrics)))

        if self.dataset.h_params['target_type'] == 'float':
            rms = {k:np.mean(v) for k, v in rmss.items()}
            rms.update({k + '_std':np.std(v) for k, v in rmss.items()})
            print("""Validation set:
                  Corr : {:.3f} +/- {:.3f}.
                  R^2: {:.3f} +/- {:.3f}""".format(
                  rms['cc'], rms['cc_std'], rms['r2'], rms['r2_std']))
            self.meta.update(results=rms)
        else:
            rms = None

        print("""{} with {} fold(s) completed. \n
              Validation Performance:
              Loss: {:.4f} +/- {:.4f}.
              Metric: {:.4f} +/- {:.4f}"""
              .format(mode, n_folds,
                      np.mean(self.cv_losses), np.std(self.cv_losses),
                      np.mean(self.cv_metrics), np.std(self.cv_metrics)))

        if len(self.dataset.h_params['test_paths']) > 0:
            print("""\n
              Test Performance:
              Loss: {:.4f} +/- {:.4f}.
              Metric: {:.4f} +/- {:.4f}"""
              .format(np.mean(self.cv_test_losses),
                      np.std(self.cv_test_losses),
                      np.mean(self.cv_test_metrics),
                      np.std(self.cv_test_metrics)))

        self.meta.train_params.update({"trained":True})
        self.update_log(rms=rms, prefix=mode)
        self.save()
        #return self.cv_losses, self.cv_metrics


    def prune_weights(self, increase_regularization=3.):
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=1e-6,
                                                      patience=10,
                                                      restore_best_weights=True)
        self.rate = 0
        self.specs["l1_lambda"] *= increase_regularization
        self.specs["l2_lambda"] *= increase_regularization
        print('Pruning weights')
        self.t_hist_p = self.km.fit(self.dataset.train,
                               validation_data=self.dataset.val,
                               epochs=30, steps_per_epoch=self.meta.train_params['eval_step'],
                               shuffle=True,
                               validation_steps=self.dataset.validation_steps,
                               callbacks=[stop_early], verbose=2)

    def shuffle_weights(self):
        print("Re-shuffling weights between folds")
        weights = self.km.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self.km.set_weights(weights)


    def plot_hist(self):
        """Plot loss history during training."""
        plt.plot(self.t_hist.history['loss'])
        plt.plot(self.t_hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def _confusion_matrix(self, y_true, y_pred):
        """Compute unnormalizewd confusion matrix"""
        y_p = _onehot(np.argmax(y_pred,1), n_classes=self.y_shape[-1])
        cm = np.dot(y_p.T, y_true)
        return cm

    def update_results(self):
        """Add training results to training log"""
        results = dict()
        results['v_metric'] = np.mean(self.cv_metrics)
        results['v_loss'] = np.mean(self.cv_losses)
        results['cv_metrics'] = self.cv_metrics
        results['cv_losses'] = self.cv_losses
        results['cv_metric_pvalues'] = self.cv_metric_pvalues

        tr_loss, tr_metric = self.evaluate(self.dataset.train)
        results['tr_metric'] = tr_metric
        results['tr_loss'] = tr_loss


        if len(self.dataset.h_params['test_paths']) > 0:
            t_loss = np.mean(self.cv_test_losses)
            t_metric = np.mean(self.cv_test_metrics)
            if self.dataset.h_params['target_type'] == 'float':
                y_true, y_pred = self.predict(self.dataset.h_params['test_paths'])
                rms_test = regression_metrics(y_true, y_pred)
                print("Test set: Corr =", rms_test['cc'], "R2 =", rms_test['r2'])
                results.update({'test_'+k:v for k,v in rms_test.items()})
            results['test_metric'] = t_metric
            results['test_loss'] = t_loss
            results['test_metrics'] = self.cv_test_metrics
            results['test_losses'] = self.cv_test_losses

        else:
            results['test_metric'] = "NA"
            results['test_loss'] = "NA"
            results['test_metrics'] = "NA"
            results['test_losses'] = "NA"

        results['cm'] = self.cm

        self.meta.update(results=results)

    def permutation_p_value(self, dataset=None, n_perm=1000):
        perm_losses = []
        perm_metrics = []
        if not dataset:
            dataset = self.dataset.val
        y_true, y_pred_obs = self.predict(dataset)
        obs_loss, obs_metric = self.evaluate(dataset)
        n = y_true.shape[0]
        for i in range(n_perm):
            shuffle = np.random.permutation(n)
            y_surrogate = y_true[shuffle, :]
            perm_losses.append(self.km.loss(y_surrogate, y_pred_obs).numpy())
            perm_metrics.append(self.km.metrics[-1](y_surrogate, y_pred_obs).numpy())
        loss_pvalue = np.sum(np.array(perm_losses) < obs_loss)/n_perm
        metric_pvalue = np.sum(np.array(perm_metrics) > obs_metric)/n_perm
        #print("Loss p-value={:.4f}".format(loss_pvalue))
        print("Metric p-value={:.4f}".format(metric_pvalue))
        return metric_pvalue

    def update_log(self, rms=None, prefix=''):
        """Logs experiment to self.model_path + self.scope + '_log.csv'.

        If the file exists, appends a line to the existing file.
        """
        savepath = os.path.join(self.model_path, self.scope + '_log.csv')
        appending = os.path.exists(savepath)
        self.update_results()

        log = dict()
        if rms:
            log.update({prefix+k:v for k,v in rms.items()})

        #dataset info
        data_dict = self.meta.data.copy()
        _ = data_dict.pop('folds')
        _ = data_dict.pop('test_fold')
        _ = data_dict.pop('train_paths')
        _ = data_dict.pop('test_paths')
        log['data_id'] = data_dict.pop('data_id')

        #results info
        results_dict = self.meta.results.copy()
        log['train metric'] = results_dict.pop('tr_metric')
        log['validation metric'] = results_dict.pop('v_metric')
        log['test metric'] =  results_dict.pop('test_metric')
        log['train loss'] = results_dict.pop('tr_loss')
        log['validation loss'] = results_dict.pop('v_loss')
        log['test loss'] =  results_dict.pop('test_loss')


        #format specs: architecture and regularization
        specs_dict = self.meta.model_specs.copy()
        specs_dict['l1_scope'] = '-'.join(self.meta.model_specs['l1_scope'])
        specs_dict['l2_scope'] = '-'.join(self.meta.model_specs['l2_scope'])
        specs_dict['unitnorm_scope'] = '-'.join(self.meta.model_specs['unitnorm_scope'])
        _ = specs_dict.pop('model_path')
        if isinstance(specs_dict['nonlin'], Callable):
            specs_dict['nonlin'] = specs_dict['nonlin'].__name__

        log.update(specs_dict)
        #training paramters
        log.update(self.meta.train_params)
        log.update(data_dict)
        log.update(results_dict)


        self.log.update(log)

        with open(savepath, 'a+', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.log.keys())
            if not appending:
                writer.writeheader()
            writer.writerow(self.log)
            print("Saving updated log to: ",  savepath)

    def save(self):
        """
        Saves the model and (optionally, patterns, confusion matrices)
        """

        self.update_results()
        weights = {k: np.stack(self.cv_weights[k], -1) for k in self.cv_weights.keys()}

        #Update and save meta file
        self.meta.update(data=self.dataset.h_params,
                         model_specs=self.specs,
                         patterns=self.cv_patterns,
                         weights=weights)

        #save the model
        self.km.save(os.path.join(self.model_path, self.model_name + '.h5'))
        if hasattr(self, 'km_enc'):
            self.km_enc.save(os.path.join(self.model_path, self.model_name + 'encoder_.h5'))


    def predict_sample(self, x):
        n_ch = self.dataset.h_params['n_ch']
        n_t = self.dataset.h_params['n_t']
        assert x.shape[-2:] == (n_t, n_ch),  "Shape mismatch! Expected {}x{}, \
            got {}x{}".format(n_t, n_ch, x.shape[-2], x.shape[-1])

        while x.ndim < 4:
            x = np.expand_dims(x, 0)

        out = self.km.predict(x)
        if self.dataset.h_params['target_type'] == 'int':
            out = np.argmax(out, -1)

        return out

    def predict(self, dataset=None, n_batches=1):
        """
        Returns
        -------
        y_true : np.array
                ground truth labels taken from the dataset

        y_pred : np.array
                model predictions
        """
        if not dataset:
            print("No dataset specified using validation dataset (Default)")
            dataset = self.dataset.val
        elif isinstance(dataset, str) or isinstance(dataset, (list, tuple)):
            dataset = self.dataset._build_dataset(dataset,
                                                 split=False,
                                                 test_batch=None,
                                                 repeat=True)
        elif not isinstance(dataset, tf.data.Dataset):
            print("Specify dataset")
            return None, None

        X = []
        y = []
        for batch_idx, (x, y_) in enumerate(dataset):
            if batch_idx >= n_batches:
                break


            X.append(x)
            y.append(y_)

        y_pred = self.km.predict(np.concatenate(X))
        y_true = np.concatenate(y)

        return y_true, y_pred

    def evaluate(self, dataset=False):
        """
        Returns
        -------
        losses : list
                model loss on a specified dataset

        metrics : np.array
                metrics evaluated on a specified dataset
        """

        if not dataset:
            print("No dataset specified using validation dataset (Default)")
            dataset = self.dataset.val
        elif isinstance(dataset, str) or isinstance(dataset, (list, tuple)):
            dataset = self.dataset._build_dataset(dataset,
                                             split=False,
                                             test_batch=None,
                                             repeat=True)
        elif not isinstance(dataset, tf.data.Dataset):
            print("Specify dataset")
            return None, None

        losses, metrics = self.km.evaluate(dataset,
                                           steps=self.dataset.validation_steps,
                                           verbose=0)
        return  losses, metrics

class SourceNet(BaseModel):
    """SourceNet

    For details see [1].

    References
    ----------
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None, specs_prefix=False):
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
            Convolution padding. Defaults to 'SAME'.}"""
        self.scope = 'varcnn'
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0)
        meta.model_specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        super(SourceNet, self).__init__(meta, dataset, specs_prefix)

    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        # self.tconv = VARConv(size=self.specs['n_latent'],
        #                      nonlin=self.specs['nonlin'],
        #                      filter_length=self.specs['filter_length'],
        #                      padding=self.specs['padding'],
        #                      specs=self.specs
        #                      )(self.inputs)
        self.tconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size = (1, self.specs['filter_length']),
            #strides=1,
            padding='same',
            depth_multiplier=self.specs['n_latent'],
            data_format='channels_first',
            dilation_rate=(1, 1),
            activation=self.specs['nonlin'],
            use_bias=True,
            depthwise_initializer='glorot_uniform',
            bias_initializer='zeros',
            depthwise_regularizer=tf.keras.regularizers.l1(self.specs['l1_lambda']),
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            bias_constraint=None,
            )(self.inputs)
        print('tconv: ', self.tconv.shape )
        self.pooled = TempPooling(pooling=self.specs['pooling'],
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['stride'],
                                  padding=self.specs['padding'],
                                  )(self.tconv)

        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=self.specs['nonlin'],
                            axis=3, specs=self.specs)(self.pooled)

        self.dmx1 = DeMixing(size=self.specs['n_latent'], nonlin=self.specs['nonlin'],
                            axis=1, specs=self.specs)(self.dmx)




        dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(self.dmx1)

        # fc1 = FullyConnected(size=self.specs['n_latent'],
        #                      nonlin=self.specs['nonlin'],
        #                      specs=self.specs)(dropout)

        self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)

        y_pred = self.fin_fc(dropout)

        return y_pred



class VARCNN(BaseModel):
    """VAR-CNN.

    For details see [1].

    References
    ----------
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None, specs_prefix=False):
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
            Convolution padding. Defaults to 'SAME'.}"""
        self.scope = 'varcnn'
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0)
        meta.model_specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        super(VARCNN, self).__init__(meta, dataset, specs_prefix)

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
                            axis=3, specs=self.specs)(self.inputs)


        self.tconv = VARConv(size=self.specs['n_latent'],
                             nonlin=self.specs['nonlin'],
                             filter_length=self.specs['filter_length'],
                             padding=self.specs['padding'],
                             specs=self.specs
                             )(self.dmx)

        self.pooled = TempPooling(pooling=self.specs['pooling'],
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['stride'],
                                  padding=self.specs['padding'],
                                  )(self.tconv)

        dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(self.pooled)

        #fc1 = FullyConnected(size=128, nonlin=tf.nn.elu,
        #                    specs=self.specs)(dropout)

        self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)

        y_pred = self.fin_fc(dropout)

        return y_pred



class FBCSP_ShallowNet(BaseModel):
    """
    Shallow ConvNet model from [2a]_.
    References
    ----------
    .. [2a] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    def __init__(self, meta, dataset=None, specs_prefix=False):
        self.scope = 'fbcsp-ShallowNet'
        meta.model_specs.setdefault('filter_length', 25)
        meta.model_specs.setdefault('n_latent', 40)
        meta.model_specs.setdefault('pooling', 75)
        meta.model_specs.setdefault('stride', 15)
        meta.model_specs.setdefault('pool_type', 'avg')
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 3e-2)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', ['conv', 'fc'])

        meta.model_specs.setdefault('unitnorm_scope', [])
        #specs.setdefault('model_path', os.path.join(self.dataset.h_params['path'], 'models'))
        super(FBCSP_ShallowNet, self).__init__(meta, dataset, specs_prefix)

    def build_graph(self):

        """Temporal conv_1 25 10x1 kernels"""
        #(self.inputs)
        inputs = tf.transpose(self.inputs,[0,3,2,1])
        #print(inputs.shape)
        #df = "channels_first"
        tconv1 = DepthwiseConv2D(
                        kernel_size=(1, self.specs['filter_length']),
                        depth_multiplier = self.specs['n_latent'],
                        strides=1,
                        padding="VALID",
                        activation = tf.identity,
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda'])
                        #kernel_constraint="maxnorm"
                        )

        tconv1_out = tconv1(inputs)
        print('tconv1: ', tconv1_out.shape) #should be n_batch, sensors, times, kernels

        sconv1 = Conv2D(filters=self.specs['n_latent'],
                        kernel_size=(self.dataset.h_params['n_ch'], 1),
                        strides=1,
                        padding="VALID",
                        activation = tf.square,
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


        sconv1_out = sconv1(tconv1_out)
        print('sconv1:',  sconv1_out.shape)

        pool1 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(sconv1_out)

        print('pool1: ', pool1.shape)
        fc_out = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)
        y_pred = fc_out(tf.keras.backend.log(pool1))
        return y_pred
#
#
class LFLSTM(BaseModel):
    # TODO! Gabi: check that the description describes the model
    """LF-CNN-LSTM

    For details see [1].

    Parameters
    ----------
    n_latent : int
        number of latent components
        Defaults to 32

    filter_length : int
        length of spatio-temporal kernels in the temporal
        convolution layer. Defaults to 7

    stride : int
        stride of the max pooling layer. Defaults to 1

    pooling : int
        pooling factor of the max pooling layer. Defaults to 2

    References
    ----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None, specs_prefix=False):
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
        Stride of the max pooling layer. Defaults to 1.
        """
        #self.scope = 'lflstm'
        self.scope = 'lf-cnn-lstm'
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 0.)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs['scope'] = self.scope
        meta.model_specs.setdefault('unitnorm_scope', [])
        #specs.setdefault('model_path',  self.dataset.h_params['save_path'])
        super(LFLSTM, self).__init__(meta, dataset, specs_prefix)


    def build_graph(self):

        self.return_sequence = True
        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
                            axis=3, specs=self.specs)
        dmx = self.dmx(self.inputs)
        #dmx = tf.reshape(dmx, [-1, self.dataset.h_params['n_t'],
        #                       self.specs['n_latent']])
        #dmx = tf.expand_dims(dmx, -1)
        print('dmx-sqout:', dmx.shape)

        self.tconv1 = LFTConv(scope="conv",
                              size=self.specs['n_latent'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
#                              stride=self.specs['stride'],
#                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        features = self.tconv1(dmx)
        pool1 = TempPooling(stride=self.specs['stride'],
                            pooling=self.specs['pooling'],
                            padding='SAME',
                            pool_type='max')


        pooled = pool1(features)
        print('features:', pooled.shape)

        fshape = tf.multiply(pooled.shape[2], pooled.shape[3])

        ffeatures = tf.reshape(pooled,
                              [-1, self.dataset.h_params['n_seq'], fshape])
        #  features = tf.expand_dims(features, 0)
        #l1_lambda = self.optimizer.params['l1_lambda']
        print('flat features:', ffeatures.shape)
        self.lstm = LSTM(scope="lstm",
                           size=self.specs['n_latent'],
                           kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal',
                           recurrent_regularizer=k_reg.l1(self.specs['l1_lambda']),
                           kernel_regularizer=k_reg.l2(self.specs['l2_lambda']),
                           bias_regularizer=None,
                           # activity_regularizer= regularizers.l1(0.01),
                           # kernel_constraint= constraints.UnitNorm(axis=0),
                           # recurrent_constraint= constraints.NonNeg(),
                           # bias_constraint=None,
                           dropout=0.1, recurrent_dropout=0.1,
                           nonlin=tf.identity,
                           unit_forget_bias=False,
                           return_sequences=self.return_sequence,
                           unroll=False)

        self.lstm_out = self.lstm(ffeatures)
        print('lstm_out:', self.lstm_out.shape)

        if self.return_sequence == True:
            self.fin_fc = DeMixing(size=self.out_dim,
                                   nonlin=tf.identity, axis=2)
        else:
            self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                                specs=self.specs)

        y_pred = self.fin_fc(self.lstm_out)
        print("fin fc out:", y_pred.shape)
        return y_pred
#
#


class Deep4(BaseModel):
    """
    Deep ConvNet model from [2b]_.
    References
    ----------
    .. [2b] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    def __init__(self, meta, dataset=None, specs_prefix=False):
        self.scope = 'deep4'
        meta.model_specs.setdefault('filter_length', 10)
        meta.model_specs.setdefault('n_latent', 25)
        meta.model_specs.setdefault('pooling', 3)
        meta.model_specs.setdefault('stride', 3)
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('nonlin', tf.nn.elu)
        meta.model_specs.setdefault('l1_lambda', 0)
        meta.model_specs.setdefault('l2_lambda', 0)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        #specs.setdefault('model_path', os.path.join(self.dataset.h_params['path'], 'models'))
        super(Deep4, self).__init__(meta, dataset, specs_prefix)

    def build_graph(self):
        self.scope = 'deep4'

        inputs = tf.keras.ops.transpose(self.inputs,[0,3,2,1])

        tconv1 = DepthwiseConv2D(
                        kernel_size=(1, self.specs['filter_length']),
                        depth_multiplier = self.specs['n_latent'],
                        strides=1,
                        padding=self.specs['padding'],
                        activation = tf.identity,
                        depthwise_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        depthwise_regularizer=k_reg.l2(self.specs['l2_lambda'])
                        #kernel_constraint="maxnorm"
                        )
        tconv1_out = tconv1(inputs)
        print('tconv1: ', tconv1_out.shape) #should be n_batch, sensors, times, kernels

        sconv1 = Conv2D(filters=self.specs['n_latent'],
                        kernel_size=(self.dataset.h_params['n_ch'], 1),
                        strides=1,
                        padding=self.specs['padding'],
                        activation=self.specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))
        sconv1_out = sconv1(tconv1_out)
        print('sconv1:',  sconv1_out.shape)

        pool1 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(sconv1_out)

        print('pool1: ', pool1.shape)

        ############################################################

        tsconv2 = Conv2D(filters=self.specs['n_latent']*2,
                        kernel_size=(1, self.specs['filter_length']),
                        strides=1,
                        padding=self.specs['padding'],
                        activation=self.specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


        tsconv2_out = tsconv2(pool1)
        print('tsconv2:',  tsconv2_out.shape)

        pool2 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(tsconv2_out)

        print('pool2: ', pool2.shape)


        ############################################################

        tsconv3 = Conv2D(filters=self.specs['n_latent']*4,
                        kernel_size=(1, self.specs['filter_length']),
                        strides=1,
                        padding=self.specs['padding'],
                        activation=self.specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


        tsconv3_out = tsconv3(pool2)
        print('tsconv3:',  tsconv3_out.shape)

        pool3 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(tsconv3_out)

        print('pool3: ', pool3.shape)

        ############################################################

        tsconv4 = Conv2D(filters=self.specs['n_latent']*8,
                        kernel_size=(1, self.specs['filter_length']),
                        strides=1,
                        padding=self.specs['padding'],
                        activation=self.specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


        tsconv4_out = tsconv4(pool3)
        print('tsconv4:',  tsconv4_out.shape)

        pool4 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(tsconv4_out)

        print('pool4: ', pool4.shape)


        fc_out = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)
        y_pred = fc_out(pool4)
        return y_pred
#
#

class EEGNet(BaseModel):
    """EEGNet.

    Parameters
    ----------
    specs : dict

        n_latent : int
            Number of (temporal) convolution kernrels in the first layer.
            Defaults to 8

        filter_length : int
            Length of temporal filters in the first layer.
            Defaults to 32

        stride : int
            Stride of the average polling layers. Defaults to 4.

        pooling : int
            Pooling factor of the average polling layers. Defaults to 4.

        dropout : float
            Dropout coefficient.

    References
    ----------
    [3] V.J. Lawhern, et al., EEGNet: A compact convolutional neural
    network for EEG-based brain–computer interfaces 10 J. Neural Eng.,
    15 (5) (2018), p. 056013

    [4] Original EEGNet implementation by the authors can be found at
    https://github.com/vlawhern/arl-eegmodels
    """
    def __init__(self, meta, dataset=None, specs_prefix=False):
        self.scope = 'eegnet8'
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs.setdefault('filter_length', 64)
        meta.model_specs.setdefault('depth_multiplier', 2)
        meta.model_specs.setdefault('n_latent', 8)
        meta.model_specs.setdefault('pooling', 4)
        meta.model_specs.setdefault('stride', 4)
        meta.model_specs.setdefault('dropout', 0.1)
        meta.model_specs.setdefault('padding', 'same')
        meta.model_specs.setdefault('nonlin', 'elu')
        meta.model_specs['scope'] = self.scope
        super(EEGNet, self).__init__(meta, dataset, specs_prefix)


    def build_graph(self):


        inputs = tf.transpose(self.inputs,[0,3,2,1])

        dropoutType = Dropout

        block1       = Conv2D(self.specs['n_latent'],
                              (1, self.specs['filter_length']),
                              padding = self.specs['padding'],
                              input_shape = (1, self.dataset.h_params['n_ch'],
                                             self.dataset.h_params['n_t']),
                              use_bias = False)(inputs)
        block1       = BatchNormalization(axis = 1)(block1)
        #print("Batchnorm:", block1.shape)
        block1       = DepthwiseConv2D((self.dataset.h_params['n_ch'], 1),
                                       use_bias = False,
                                       depth_multiplier = self.specs['depth_multiplier'],
                                       depthwise_constraint = constraints.MaxNorm(1.))(block1)
        #block1       = BatchNormalization(axis = 1)(block1)
        block1       = layers.Activation(self.specs['nonlin'])(block1)
        block1       = layers.AveragePooling2D((1, self.specs['pooling']))(block1)
        print("Block 1:", block1.shape)
        block1       = dropoutType(self.specs['dropout'])(block1)

        block2       = SeparableConv2D(self.specs['n_latent']*self.specs['depth_multiplier'], (1, self.specs['filter_length']//self.specs["pooling"]),
                                       use_bias = False, padding = self.specs['padding'])(block1)
        #block2       = BatchNormalization(axis = 1)(block2)

        #print("Batchnorm 2:", block2.shape)

        block2       = layers.Activation(self.specs['nonlin'])(block2)
        block2       = layers.AveragePooling2D((1, self.specs['pooling']*2))(block2)
        block2       = dropoutType(self.specs['dropout'])(block2)
        print("Block 2:", block2.shape)

        fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)
        y_pred = fin_fc(block2)

        return y_pred


# class SourceNet(BaseModel):
#     """

#     """
#     def __init__(self, meta, dataset=None, specs_prefix=False):
#         self.scope = 'SourceNet'
#         meta.model_specs.setdefault('unitnorm_scope', [])
#         meta.model_specs.setdefault('filter_length', 10)
#         meta.model_specs.setdefault('n_latent', 25)
#         meta.model_specs.setdefault('pooling', 3)
#         meta.model_specs.setdefault('stride', 3)
#         meta.model_specs.setdefault('pool_type', 'max')
#         meta.model_specs.setdefault('padding', 'SAME')
#         meta.model_specs.setdefault('nonlin', tf.nn.elu)
#         meta.model_specs.setdefault('l1_lambda', 0)
#         meta.model_specs.setdefault('l2_lambda', 0)
#         meta.model_specs.setdefault('l1_scope', [])
#         meta.model_specs.setdefault('l2_scope', [])
#         meta.model_specs.setdefault('unitnorm_scope', [])
#         #specs.setdefault('model_path', os.path.join(self.dataset.h_params['path'], 'models'))
#         super(SourceNet, self).__init__(meta, dataset, specs_prefix)

#     def build_graph(self):
#         self.scope = 'SourceNet'

#         self.scope = 'deep4'

#         inputs = tf.keras.ops.transpose(self.inputs,[0,3,2,1])

#         tconv1 = DepthwiseConv2D(
#                         kernel_size=(1, self.specs['filter_length']),
#                         depth_multiplier = self.specs['n_latent'],
#                         strides=1,
#                         padding=self.specs['padding'],
#                         activation = tf.identity,
#                         depthwise_initializer="he_uniform",
#                         bias_initializer=Constant(0.1),
#                         data_format="channels_last",
#                         depthwise_regularizer=k_reg.l2(self.specs['l2_lambda'])
#                         #kernel_constraint="maxnorm"
#                         )
#         tconv1_out = tconv1(inputs)
#         print('tconv1: ', tconv1_out.shape) #should be n_batch, sensors, times, kernels

#         sconv1 = Conv2D(filters=self.specs['n_latent'],
#                         kernel_size=(self.dataset.h_params['n_ch'], 1),
#                         strides=1,
#                         padding=self.specs['padding'],
#                         activation=self.specs['nonlin'],
#                         kernel_initializer="he_uniform",
#                         bias_initializer=Constant(0.1),
#                         data_format="channels_last",
#                         #data_format="channels_first",
#                         kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))
#         sconv1_out = sconv1(tconv1_out)
#         print('sconv1:',  sconv1_out.shape)

#         pool1 = TempPooling(pooling=self.specs['pooling'],
#                                   pool_type="avg",
#                                   stride=self.specs['stride'],
#                                   padding='SAME',
#                                   )(sconv1_out)

#         print('pool1: ', pool1.shape)

#         ############################################################

#         tsconv2 = Conv2D(filters=self.specs['n_latent'],
#                         kernel_size=(1, self.specs['filter_length']),
#                         strides=1,
#                         padding=self.specs['padding'],
#                         activation=self.specs['nonlin'],
#                         kernel_initializer="he_uniform",
#                         bias_initializer=Constant(0.1),
#                         data_format="channels_last",
#                         #data_format="channels_first",
#                         kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


#         tsconv2_out = tsconv2(pool1)
#         print('tsconv2:',  tsconv2_out.shape)

#         pool2 = TempPooling(pooling=self.specs['pooling'],
#                                   pool_type="avg",
#                                   stride=self.specs['stride'],
#                                   padding='SAME',
#                                   )(tsconv2_out)

#         print('pool2: ', pool2.shape)

#         dmx1 = DeMixing(size=4, nonlin=tf.identity,
#                             axis=1, specs=self.specs)(pool2)
#         print('dmx1: ', dmx1.shape)

#         dmx2 = DeMixing(size=4, nonlin=tf.identity,
#                             axis=2, specs=self.specs)(dmx1)
#         print('dmx2: ', dmx2.shape)

#         # dmx1 = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
#         #                     axis=1, specs=self.specs)(pool2)
#         # print('dmx1: ', dmx1.shape)
#         ############################################################

#         # tsconv3 = Conv2D(filters=self.specs['n_latent'],
#         #                 kernel_size=(1, self.specs['filter_length']),
#         #                 strides=1,
#         #                 padding=self.specs['padding'],
#         #                 activation=self.specs['nonlin'],
#         #                 kernel_initializer="he_uniform",
#         #                 bias_initializer=Constant(0.1),
#         #                 data_format="channels_last",
#         #                 #data_format="channels_first",
#         #                 kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


#         # tsconv3_out = tsconv3(pool2)
#         # print('tsconv3:',  tsconv3_out.shape)

#         # pool3 = TempPooling(pooling=self.specs['pooling'],
#         #                           pool_type="avg",
#         #                           stride=self.specs['stride'],
#         #                           padding='SAME',
#         #                           )(tsconv3_out)

#         #print('pool3: ', pool3.shape)




#         fc_out = FullyConnected(size=self.out_dim, nonlin=tf.identity,
#                             specs=self.specs)
#         y_pred = fc_out(dmx2)
#         return y_pred
# class SimpleNet(LFCNN):
#     """
#         Petrosyan, A., Sinkin, M., Lebedev, M. A., & Ossadtchi, A.  Decoding and interpreting cortical signals with
#         a compact convolutional neural network, 2021, Journal of Neural Engineering, 2021,
#         https://doi.org/10.1088/1741-2552/abe20e
#     """
#     def __init__(self, Dataset, specs=None):
#         if specs is None:
#             specs=dict()
#         super().__init__(Dataset, specs)

#     def build_graph(self):
#         self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
#                             axis=3, specs=self.specs)
#         self.dmx_out = self.dmx(self.inputs)

#         self.tconv = LFTConv(
#             size=self.specs['n_latent'],
#             nonlin=self.specs['nonlin'],
#             filter_length=self.specs['filter_length'],
#             padding=self.specs['padding'],
#             specs=self.specs
#         )
#         self.tconv_out = self.tconv(self.dmx_out)

#         self.envconv = LFTConv(
#             size=self.specs['n_latent'],
#             nonlin=self.specs['nonlin'],
#             filter_length=self.specs['filter_length'],
#             padding=self.specs['padding'],
#             specs=self.specs
#         )

#         self.envconv_out = self.envconv(self.tconv_out)
#         self.pool = lambda X: X[:, :, ::self.specs['pooling'], :]

#         self.pooled = self.pool(self.envconv_out)

#         dropout = Dropout(
#             self.specs['dropout'],
#             noise_shape=None
#         )(self.pooled)

#         self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
#                             specs=self.specs)

#         y_pred = self.fin_fc(dropout)

#         return y_pred

#     def compute_patterns(self, data_path=None, *, output='patterns'):

#         if not data_path:
#             print("Computing patterns: No path specified, using validation dataset (Default)")
#             ds = self.dataset.val
#         elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
#             ds = self.dataset._build_dataset(
#                 data_path,
#                 split=False,
#                 test_batch=None,
#                 repeat=True
#             )
#         elif isinstance(data_path, Dataset):
#             if hasattr(data_path, 'test'):
#                 ds = data_path.test
#             else:
#                 ds = data_path.val
#         elif isinstance(data_path, tf.data.Dataset):
#             ds = data_path
#         else:
#             raise AttributeError('Specify dataset or data path.')

#         X, y = [row for row in ds.take(1)][0]

#         self.out_w_flat = self.fin_fc.w.numpy()
#         self.out_weights = np.reshape(
#             self.out_w_flat,
#             [-1, self.dmx.size, self.out_dim]
#         )
#         self.out_biases = self.fin_fc.b.numpy()
#         self.feature_relevances = self.componentwise_loss(X, y)
#         self.branchwise_loss(X, y)

#         # compute temporal convolution layer outputs for vis_dics
#         tc_out = self.pool(self.tconv(self.dmx(X)).numpy())

#         # compute data covariance
#         X = X - tf.reduce_mean(X, axis=-2, keepdims=True)
#         X = tf.transpose(X, [3, 0, 1, 2])
#         X = tf.reshape(X, [X.shape[0], -1])
#         self.dcov = tf.matmul(X, tf.transpose(X))

#         # get spatial extraction fiter weights
#         demx = self.dmx.w.numpy()

#         kern = np.squeeze(self.tconv.filters.numpy()).T

#         X = X.numpy().T

#         patterns = []
#         X_filt = np.zeros_like(X)
#         for i_comp in range(kern.shape[0]):
#             for i_ch in range(X.shape[1]):
#                 x = X[:, i_ch]
#                 X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
#             patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
#         self.patterns = np.array(patterns).T


#         if 'patterns' in output:
#             if 'old' in output:
#                 self.patterns = np.dot(self.dcov, demx)
#             else:
#                 patterns = []
#                 X_filt = np.zeros_like(X)
#                 for i_comp in range(kern.shape[0]):
#                     for i_ch in range(X.shape[1]):
#                         x = X[:, i_ch]
#                         X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
#                     patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
#                 self.patterns = np.array(patterns).T
#         else:
#             self.patterns = demx

#         self.lat_tcs = np.dot(demx.T, X.T)

#         del X

#         #  Temporal conv stuff
#         self.filters = kern.T
#         self.tc_out = np.squeeze(tc_out)
#         self.corr_to_output = self.get_output_correlations(y)

#     def plot_patterns(
#         self, sensor_layout=None, sorting='l2', percentile=90,
#         scale=False, class_names=None, info=None
#     ):
#         order, ts = self._sorting(sorting)
#         self.uorder = order.ravel()
#         l_u = len(self.uorder)
#         if info:
#             info.__setstate__(dict(_unlocked=True))
#             info['sfreq'] = 1.
#             self.fake_evoked = evoked.EvokedArray(self.patterns, info, tmin=0)
#             if l_u > 1:
#                 self.fake_evoked.data[:, :l_u] = self.fake_evoked.data[:, self.uorder]
#             elif l_u == 1:
#                 self.fake_evoked.data[:, l_u] = self.fake_evoked.data[:, self.uorder[0]]
#             self.fake_evoked.crop(tmax=float(l_u))
#             if scale:
#                 _std = self.fake_evoked.data[:, :l_u].std(0)
#                 self.fake_evoked.data[:, :l_u] /= _std
#         elif sensor_layout:
#             lo = channels.read_layout(sensor_layout)
#             info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
#             orig_xy = np.mean(lo.pos[:, :2], 0)
#             for i, ch in enumerate(lo.names):
#                 if info['chs'][i]['ch_name'] == ch:
#                     info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/3.
#                     #info['chs'][i]['loc'][4:] = 0
#                 else:
#                     print("Channel name mismatch. info: {} vs lo: {}".format(
#                         info['chs'][i]['ch_name'], ch))

#             self.fake_evoked = evoked.EvokedArray(self.patterns, info)

#             if l_u > 1:
#                 self.fake_evoked.data[:, :l_u] = self.fake_evoked.data[:, self.uorder]
#             elif l_u == 1:
#                 self.fake_evoked.data[:, l_u] = self.fake_evoked.data[:, self.uorder[0]]
#             self.fake_evoked.crop(tmax=float(l_u))
#             if scale:
#                 _std = self.fake_evoked.data[:, :l_u].std(0)
#                 self.fake_evoked.data[:, :l_u] /= _std
#         else:
#             raise ValueError("Specify sensor layout")


#         if np.any(self.uorder):
#             nfilt = max(self.out_dim, 8)
#             nrows = max(1, l_u//nfilt)
#             ncols = min(nfilt, l_u)
#             f, ax = plt.subplots(nrows, ncols, sharey=True)
#             plt.tight_layout()
#             f.set_size_inches([16, 3])
#             ax = np.atleast_2d(ax)

#             for ii in range(nrows):
#                 fake_times = np.arange(ii * ncols,  (ii + 1) * ncols, 1.)
#                 vmax = np.percentile(self.fake_evoked.data[:, :l_u], 95)
#                 self.fake_evoked.plot_topomap(
#                     times=fake_times,
#                     axes=ax[ii],
#                     colorbar=False,
#                     vmax=vmax,
#                     scalings=1,
#                     time_format="Branch #%g",
#                     title='Patterns ('+str(sorting)+')',
#                     outlines='head',
#                 )

#     def branchwise_loss(self, X, y):
#         model_weights_original = self.km.get_weights().copy()
#         base_loss, _ = self.km.evaluate(X, y, verbose=0)

#         losses = []
#         for i in range(self.specs["n_latent"]):
#             model_weights = model_weights_original.copy()
#             spatial_weights = model_weights[0].copy()
#             spatial_biases = model_weights[1].copy()
#             temporal_biases = model_weights[3].copy()
#             env_biases = model_weights[5].copy()
#             spatial_weights[:, i] = 0
#             spatial_biases[i] = 0
#             temporal_biases[i] = 0
#             env_biases[i] = 0
#             model_weights[0] = spatial_weights
#             model_weights[1] = spatial_biases
#             model_weights[3] = temporal_biases
#             model_weights[5] = env_biases
#             self.km.set_weights(model_weights)
#             losses.append(self.km.evaluate(X, y, verbose=0)[0])
#         self.km.set_weights(model_weights_original)
#         self.branch_relevance_loss = base_loss - np.array(losses)

#     def plot_branch(
#         self,
#         branch_num: int,
#         info: Info,
#         params: Optional[list[str]] = ['input', 'output', 'response']
#     ):
#         info.__setstate__(dict(_unlocked=True))
#         info['sfreq'] = 1.
#         sorting = np.argsort(self.branch_relevance_loss)[::-1]
#         data = self.patterns[:, sorting]
#         filters = self.filters[:, sorting]
#         relevances = self.branch_relevance_loss - self.branch_relevance_loss.min()
#         relevance = sorted([np.round(rel/relevances.sum(), 2) for rel in relevances], reverse=True)[branch_num]
#         self.fake_evoked = evoked.EvokedArray(data, info, tmin=0)
#         fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
#         fig.tight_layout()

#         self.fs = self.dataset.h_params['fs']

#         out_filter = filters[:, branch_num]
#         _, psd = welch(self.lat_tcs[branch_num], fs=self.fs, nperseg=self.fs * 2)
#         w, h = (lambda w, h: (w, h))(*freqz(out_filter, 1, worN=self.fs))
#         frange = w / np.pi * self.fs / 2
#         z = lambda x: (x - x.mean())/x.std()

#         for param in params:
#             if param == 'input':
#                 finput = psd[:-1]
#                 finput = z(finput)
#                 ax2.plot(frange, finput - finput.min(), color='tab:blue')
#             elif param == 'output':
#                 foutput = np.real(finput * h * np.conj(h))
#                 foutput = z(foutput)
#                 ax2.plot(frange, foutput - foutput.min(), color='tab:orange')
#             elif param == 'response':
#                 fresponce = np.abs(h)
#                 fresponce = z(fresponce)
#                 ax2.plot(frange, fresponce - fresponce.min(), color='tab:green')
#             elif param == 'pattern':
#                 fpattern = finput * np.abs(h)
#                 fpattern = z(fpattern)
#                 ax2.plot(frange, fpattern - fpattern.min(), color='tab:pink')

#         ax2.legend([param.capitalize() for param in params])
#         ax2.set_xlim(0, 100)

#         fig.suptitle(f'Branch {branch_num}', y=0.95, x=0.2, fontsize=30)
#         fig.set_size_inches(10, 5)
#         self.fake_evoked.plot_topomap(
#             times=branch_num,
#             axes=ax1,
#             colorbar=False,
#             scalings=1,
#             time_format="",
#             outlines='head',
#         )

#         return fig

class NoisyTrainer:
    def __init__(self, model, model_path, noise_std=.1, patience=5, min_delta=0.001):
        """
        Initialize the trainer with a model, noise parameters, and early stopping configuration

        Args:
            model: Keras model
            noise_std: Standard deviation of Gaussian noise to add to labels
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored metric to qualify as an improvement
        """
        self.model = model
        self.model_path = model_path + '_best.weights.h5'
        self.noise_std = noise_std
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.metric = tf.keras.metrics.MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam()

        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    @tf.function
    def add_noise_to_labels(self, labels):
        """Add Gaussian noise to the labels"""
        noise = tf.random.normal(shape=tf.shape(labels),
                               mean=0.0,
                               stddev=self.noise_std)
        return labels + noise

    @tf.function
    def train_step(self, x, y):
        """Single training step with noisy labels"""
        # Add noise to labels
        noisy_y = self.add_noise_to_labels(y)

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x, training=True)
            # Calculate loss using noisy labels
            loss = self.loss_fn(noisy_y, predictions)

        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def validation_step(self, x, y):
        """Validation step without noise and training mode"""
        predictions = self.model(x, training=False)
        val_loss = self.loss_fn(y, predictions)
        metric = self.metric(y, predictions)
        return val_loss, metric

    def train(self, train_dataset, val_dataset=None, epochs=1000, eval_step=5,
              val_steps=1, restore_best_weights=True):
        """
        Train the model with optional validation and early stopping

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            epochs: Maximum number of training epochs

        Returns:
            Training history dictionary
        """
        training_history = {
            'train_loss': [],
            'val_loss': []
        }

        if val_dataset is not None:
            val_iter = iter(val_dataset)
        for epoch in range(epochs):
            train_iter = iter(train_dataset)

            # Train until it's time to evaluate
            train_losses = []
            for i in range(eval_step):
                x_batch, y_batch = next(train_iter)
                loss = self.train_step(x_batch, y_batch)
                train_losses.append(float(loss))

            if val_dataset is not None:
            #Evalute on validation set
                val_losses = []
                val_metrics = []
                for i in range(val_steps):
                    x_val_batch, y_val_batch = next(val_iter)
                    val_loss, val_metric = self.validation_step(x_val_batch, y_val_batch)
                    val_losses.append(float(val_loss))
                    val_metrics.append(float(val_metric))

            #Calculate output
            avg_train_loss = np.mean(train_losses)
            training_history['train_loss'].append(avg_train_loss)
            avg_val_loss = np.mean(val_losses)
            avg_val_metric = np.mean(val_metrics)
            training_history['val_loss'].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < self.best_val_loss - self.min_delta:
                #print("Eval decr")
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
                # Optionally save the best model
                if restore_best_weights:
                    self.model.save_weights(self.model_path)
            else:
                self.epochs_without_improvement += 1
                #print("Eval no decr")

            print(f"Epoch {epoch + 1}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, "
                  f"Val Metric = {avg_val_metric:.4f}")
            train_dataset.shuffle(10000)
            # Check for early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                if restore_best_weights:
                    print("Restoring best weights")
                    self.model.load_weights(self.model_path, skip_mismatch=True)
                break


        return training_history


