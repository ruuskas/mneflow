"""Tests for the connectivity models in mneflow.fc_models.

Uses small synthetic connectivity data (6 nodes, 4 frequencies) written with
produce_tfrecords, so the models are built, trained (with cross-validation
and a holdout set), logged and saved through the real pipeline.
"""

import contextlib
import csv
import io
import os

import keras
import numpy as np
import pytest

import mneflow
from mneflow.fc_models import (
    Conv3DModel,
    WeightedSum3dModel,
    SymmetricModel,
    _fold_stability,
    input_covariance,
)

N_SAMPLES, N_NODES, N_FREQS, N_FOLDS = 96, 6, 4, 3

# (model class, model specs, expected scope)
CONFIGS = {
    'conv3d': (Conv3DModel, {'n_latent': 4}, 'conv3d'),
    'weighted_sum': (
        WeightedSum3dModel,
        {'n_latent1': 4, 'n_latent2': 2},
        'weighted_sum_3d',
    ),
    'symmetric': (
        SymmetricModel,
        {'n_latent': 4, 'trainable_pointwise_kernel': True},
        'symmetric_tpk',
    ),
    'symmetric_fixed_pointwise': (
        SymmetricModel,
        {'n_latent': 4, 'trainable_pointwise_kernel': False},
        'symmetric_tpk',
    ),
}
COMMON_SPECS = dict(
    n_latent_dense=8,
    dropout=0.1,
    stddev=0.0,
    spatial_dropout=0.1,
    l1_lambda=0.0,
    l2_lambda=0.0,
    l1_scope=[],
    l2_scope=[],
    unitnorm_scope=[],
)


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def make_data(signal=0.8, seed=0):
    """Symmetric connectivity whose level encodes a sorted target."""
    rng = np.random.default_rng(seed)
    y = np.sort(rng.uniform(20, 80, N_SAMPLES))
    level = (y - 20) / 60
    X = (
        rng.random((N_SAMPLES, N_NODES, N_NODES, N_FREQS)) * (1 - signal)
        + signal * level[:, None, None, None]
    ).astype(np.float32)
    X = (X + X.transpose(0, 2, 1, 3)) / 2
    X[:, np.arange(N_NODES), np.arange(N_NODES), :] = 0
    return X, y


@pytest.fixture(scope='module')
def tfr_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('fc_tfr')
    np.random.seed(0)
    _quiet(
        mneflow.produce_tfrecords,
        make_data(),
        path=str(path),
        data_id='fc',
        fs=1,
        input_type='fconn',
        target_type='float',
        n_folds=N_FOLDS,
        overwrite=True,
        test_set='holdout',
    )
    return path


def make_model(tfr_path, name, model_id=None, train_batch=16, **extra_specs):
    cls, specs, _ = CONFIGS[name]
    keras.utils.set_random_seed(0)
    meta = mneflow.load_meta(str(tfr_path), 'fc')
    # Training saves the model specs into the shared metadata file, so
    # start every model from empty specs
    meta.model_specs = {}
    meta.update(
        model_specs=dict(
            COMMON_SPECS, **specs, **extra_specs, model_id=model_id or name
        )
    )
    dataset = mneflow.Dataset(meta, train_batch=train_batch)
    model = _quiet(cls, meta=meta, dataset=dataset, specs_prefix=False)
    _quiet(
        model.build,
        loss='MAE',
        metrics='RootMeanSquaredError',
        learn_rate=1e-2,
    )
    return model


def train_cv(model, n_epochs=2, eval_step=3, early_stopping=1):
    _quiet(
        model.train,
        n_epochs=n_epochs,
        eval_step=eval_step,
        early_stopping=early_stopping,
        mode='cv',
        collect_patterns=isinstance(model, SymmetricModel),
    )


def read_log(model):
    path = os.path.join(model.model_path, model.scope + '_log.csv')
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_list(text):
    return [
        float(x)
        for x in text.replace('np.float64(', '')
        .replace(')', '')
        .strip('[]')
        .split(',')
        if x.strip()
    ]


@pytest.mark.parametrize('name', list(CONFIGS))
def test_builds_with_expected_output_shape(tfr_path, name):
    model = make_model(tfr_path, name)
    assert model.y_pred.shape == (None, 1)


@pytest.mark.parametrize('name', list(CONFIGS))
def test_scope_is_consistent(tfr_path, name):
    """The scope names the log and saved model files, and is stored in
    the metadata; all of these must agree."""
    model = make_model(tfr_path, name)
    assert model.scope == CONFIGS[name][2]
    assert model.meta.model_specs['scope'] == model.scope
    assert model.model_name.startswith(model.scope + '_')


def test_symmetric_model_uses_attention_rate(tfr_path):
    model = make_model(tfr_path, 'symmetric', attention_rate=2)
    assert model.specs['attention_rate'] == 2
    assert 'attention_ratio' not in model.specs


@pytest.mark.parametrize('name', list(CONFIGS))
def test_cv_training_writes_log_and_saves_model(tfr_path, name):
    model = make_model(tfr_path, name, model_id=f'log_test_{name}')
    train_cv(model)
    rows = [
        r for r in read_log(model) if r['spec_model_id'] == f'log_test_{name}'
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row['loss'] == 'MAE'
    assert row['model_id'] == model.model_name
    # One validation and one holdout score per CV fold
    assert len(parse_list(row['cv_losses'])) == N_FOLDS
    assert len(parse_list(row['test_losses'])) == N_FOLDS
    assert all(np.isfinite(parse_list(row['test_losses'])))
    if name == 'weighted_sum':
        assert row['n_latent'] == ''  # this model has no n_latent
    assert os.path.exists(
        os.path.join(model.model_path, model.model_name + '.h5')
    )


def test_symmetric_patterns_and_roi_labels(tfr_path):
    model = make_model(tfr_path, 'symmetric')
    train_cv(model)
    # One collected pattern per CV fold (the holdout fold is not trained)
    dcov = model.cv_patterns['dcov']
    assert dcov.shape[-1] == N_FOLDS
    assert all(np.any(dcov[..., i]) for i in range(N_FOLDS))
    patterns = _quiet(model.extract_patterns)
    for key in ('row', 'col', 'diag'):
        assert patterns['spatial_patterns'][key].shape == (N_NODES, N_FOLDS)

    # Default return value is unchanged: a dict of label indices
    roi = _quiet(model.get_roi_labels, patterns=patterns)
    assert set(roi) == {'row', 'col', 'diag', 'row-col'}

    roi2, stability = _quiet(
        model.get_roi_labels, patterns=patterns, return_stability=True
    )
    assert set(roi2) == set(roi)
    for key in ('row', 'col', 'diag'):
        assert 0 <= stability[key]['jaccard'] <= 1
        assert 0 <= stability[key]['abs_r'] <= 1


def test_symmetric_patterns_without_trainable_pointwise(tfr_path):
    """With trainable_pointwise_kernel=False, one frequency weighting is
    shared by all components (pointwise weights (n_freqs, n_folds))."""
    model = make_model(tfr_path, 'symmetric_fixed_pointwise')
    train_cv(model)
    assert model.meta.weights['pointwise'].shape == (N_FREQS, N_FOLDS)
    pointwise = model.meta.weights['pointwise_effective']
    assert pointwise.shape == (N_FREQS, N_FOLDS)
    patterns = _quiet(model.extract_patterns)
    for key in ('row', 'col', 'diag'):
        assert patterns['spatial_patterns'][key].shape == (N_NODES, N_FOLDS)
    # The strongest frequency in each fold is that of the shared weights
    # (of the SquareSum3d output)
    np.testing.assert_array_equal(
        patterns['freq_inds']['row'], np.argmax(pointwise, 0)
    )
    np.testing.assert_array_equal(
        patterns['freq_inds']['diag'], np.argmax(np.abs(pointwise), 0)
    )
    roi = _quiet(model.get_roi_labels, patterns=patterns)
    assert set(roi) == {'row', 'col', 'diag', 'row-col'}


def test_fold_stability_measures():
    rng = np.random.default_rng(0)
    pattern = rng.normal(size=(50, 1))
    same = np.repeat(pattern, 3, axis=1)
    stab = _fold_stability(same, threshold=1.0)
    assert stab['jaccard'] == pytest.approx(1)
    assert stab['abs_r'] == pytest.approx(1)

    unrelated = rng.normal(size=(5000, 3))
    stab = _fold_stability(unrelated, threshold=1.0)
    assert stab['abs_r'] < 0.1
    assert stab['jaccard'] < 0.3


@pytest.mark.slow
@pytest.mark.parametrize('name', ['conv3d', 'weighted_sum', 'symmetric'])
def test_models_learn_target_signal(tfr_path, name):
    """On data whose connectivity level encodes the target, the holdout
    error must be clearly below predicting the mean training target
    (a model stuck at chance level scores about the same as that)."""
    model = make_model(tfr_path, name, model_id=f'learn_{name}')
    train_cv(model, n_epochs=60, eval_step=10, early_stopping=10)
    row = [
        r for r in read_log(model) if r['spec_model_id'] == f'learn_{name}'
    ][0]
    holdout_mae = np.mean(parse_list(row['test_losses']))

    _, y = make_data()
    test_fold = np.asarray(model.meta.data['test_fold'])
    train_y = np.delete(y, test_fold)
    baseline_mae = np.mean(np.abs(y[test_fold] - train_y.mean()))
    assert holdout_mae < 0.7 * baseline_mae


def test_fixed_pointwise_patterns_need_single_cross_filter(tfr_path):
    model = make_model(tfr_path, 'symmetric_fixed_pointwise', n_latent_cross=2)
    train_cv(model)
    with pytest.raises(NotImplementedError, match='n_latent_cross=1'):
        _quiet(model.extract_patterns)


def test_cv_training_stores_test_predictions(tfr_path):
    """With store_test_predictions, each fold model's predictions for the
    whole test set are kept, in test_fold order."""
    model = make_model(tfr_path, 'conv3d', model_id='pred_test')
    _quiet(
        model.train,
        n_epochs=2,
        eval_step=3,
        early_stopping=1,
        mode='cv',
        store_test_predictions=True,
    )
    _, y = make_data()
    test_fold = np.asarray(model.meta.data['test_fold'])
    assert len(model.cv_test_predictions) == N_FOLDS
    for fold in model.cv_test_predictions:
        np.testing.assert_allclose(
            np.ravel(fold['y_true']), y[test_fold], rtol=1e-5
        )
        assert np.ravel(fold['y_pred']).shape == test_fold.shape


def _produce(path, inputs, test_set='holdout'):
    np.random.seed(0)
    return _quiet(
        mneflow.produce_tfrecords,
        inputs,
        path=str(path),
        data_id='fc',
        fs=1,
        input_type='fconn',
        target_type='float',
        n_folds=N_FOLDS,
        overwrite=True,
        test_set=test_set,
    )


def test_loso_test_predictions_cover_each_subject(tmp_path):
    """With loso and subjects of different sizes, each fold's test
    predictions cover exactly the held-out subject's samples."""
    # Unequal sizes, not multiples of N_FOLDS; the first is larger than
    # the test batch (100)
    data = [make_data(seed=0), make_data(seed=1)]
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([d[1] for d in data])
    sizes = [157, 35]
    inputs = [(X[:157], y[:157]), (X[157:], y[157:])]
    _produce(tmp_path, inputs, test_set='loso')
    model = make_model(tmp_path, 'conv3d', model_id='loso_pred')
    _quiet(
        model.train,
        n_epochs=1,
        eval_step=2,
        early_stopping=1,
        mode='loso',
        store_test_predictions=True,
    )
    assert len(model.cv_test_predictions) == len(inputs)
    for (_, y_subj), size, fold in zip(
        inputs, sizes, model.cv_test_predictions
    ):
        assert np.ravel(fold['y_pred']).shape == (size,)
        np.testing.assert_allclose(
            np.sort(np.ravel(fold['y_true'])), np.sort(y_subj), rtol=1e-5
        )


def test_train_passes_shapley_order_to_collect_patterns(tfr_path):
    """Models whose collect_patterns takes shapley_order (e.g. LFCNN) get
    it from train; the connectivity models' collect_patterns does not
    take it (covered by the symmetric model tests)."""
    model = make_model(tfr_path, 'conv3d', model_id='shapley')
    calls = []

    def collect_patterns(fold=0, n_folds=1, shapley_order=0):
        calls.append(shapley_order)

    model.collect_patterns = collect_patterns
    _quiet(
        model.train,
        n_epochs=1,
        eval_step=2,
        early_stopping=1,
        mode='cv',
        collect_patterns=True,
        shapley_order=2,
    )
    assert calls == [2] * N_FOLDS


def test_log_with_old_columns_is_rewritten(tmp_path):
    """Appending to a log written with other columns (e.g. by an older
    mneflow without spec_model_id) keeps every row aligned."""
    _produce(tmp_path, make_data())
    model = make_model(tmp_path, 'conv3d', model_id='new_row')
    path = os.path.join(model.model_path, model.scope + '_log.csv')
    os.makedirs(model.model_path, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['data_id', 'model_id', 'loss'])
        writer.writeheader()
        writer.writerow({'data_id': 'old', 'model_id': 'm0', 'loss': 'MSE'})
    train_cv(model)
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = list(reader)
    assert header[:3] == ['data_id', 'model_id', 'loss']
    assert 'spec_model_id' in header
    assert all(len(line) == len(header) for line in lines)
    old, new = read_log(model)
    assert old['data_id'] == 'old' and old['loss'] == 'MSE'
    assert old['model_id'] == 'm0'
    assert old['spec_model_id'] == ''
    assert new['spec_model_id'] == 'new_row'
    assert new['model_id'] == model.model_name


@pytest.mark.parametrize('name', list(CONFIGS))
def test_no_channels_first_layers(tfr_path, name):
    """TensorFlow's CPU kernels support only channels_last convolutions
    (older versions fail on channels_first depthwise convolutions)."""
    model = make_model(tfr_path, name)
    formats = [
        getattr(layer, 'data_format', None) for layer in model.km.layers
    ]
    assert 'channels_first' not in formats


def test_trainable_pointwise_weights_each_component(tfr_path):
    """The trainable pointwise layer weights the last axis with a separate
    kernel for each component (axis 1), as a channels_first depthwise
    convolution would."""
    model = make_model(tfr_path, 'symmetric')
    rng = np.random.default_rng(0)
    inputs = rng.random((3,) + tuple(model.inputs.shape[1:]), np.float32)
    shape = tuple(model.squaresum_3d(inputs).shape)
    x = rng.normal(size=shape).astype(np.float32)
    kernel = rng.normal(size=model.pointwise.kernel.shape).astype(np.float32)
    bias = rng.normal(size=model.pointwise.bias.shape).astype(np.float32)
    model.pointwise.set_weights([kernel, bias])
    expected = np.einsum('bahw,wa->bah', x, kernel[0, :, :, 0])
    expected = np.maximum(expected + bias[None, :, None], 0)[..., None]
    got = keras.ops.convert_to_numpy(model._apply_pointwise(x))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_ablation_model_trains_on_parcel_subset(tfr_path):
    """ablation_analysis builds a model on a subset of the parcels: its
    input has the subset's size, the selected entries are X[:, idx][:, :,
    idx], and it trains in CV and predicts the whole test set even when the
    batch size does not divide the fold sizes."""
    model = make_model(tfr_path, 'symmetric', model_id='abl_full')
    idx = np.array([0, 2, 5])
    specs = dict(model.meta.model_specs, batch_size=16)
    meta_abl, model_abl, dataset = _quiet(model.ablation_analysis, 'test',
                                          idx, specs)
    assert tuple(model_abl.input_shape) == (len(idx), len(idx), N_FREQS)
    # This model's weights and patterns are not copied
    assert meta_abl.weights == {} and meta_abl.patterns == {}

    # The test data are not shuffled: compare the first batch entry by
    # entry
    x_full, _ = next(iter(model.dataset.test))
    x_sub, _ = next(iter(dataset.test))
    np.testing.assert_allclose(
        x_sub.numpy(), x_full.numpy()[:, idx][:, :, idx], rtol=1e-6
    )

    _quiet(model_abl.build, loss='MAE', metrics='RootMeanSquaredError',
           learn_rate=1e-2)
    _quiet(model_abl.train, n_epochs=2, eval_step=3, early_stopping=1,
           mode='cv', store_test_predictions=True)
    test_fold = np.asarray(meta_abl.data['test_fold'])
    _, y = make_data()
    assert len(model_abl.cv_test_predictions) == N_FOLDS
    for fold in model_abl.cv_test_predictions:
        np.testing.assert_allclose(np.ravel(fold['y_true']), y[test_fold],
                                   rtol=1e-5)



def test_row_and_col_components_are_the_strongest_weight(tfr_path):
    """'row' and 'col' are the row and column of the largest entry of the
    component-by-component output weight matrix of each fold (the row index
    used to be always 0)."""
    model = make_model(tfr_path, 'symmetric', model_id='rowcol_test')
    train_cv(model)
    weights = model.meta.weights
    n_comp = model.cv_patterns['ccms']['pointwise'].shape[0]
    # One output unit (index 0) whose only weight is the component pair
    # (n_comp - 1, 1)
    weights['out_w_flat'][:] = 0
    weights['out_w_flat'][0, :, :] = 1
    weights['fc1_w_flat'][:] = 0
    weights['fc1_w_flat'][(n_comp - 1) * n_comp + 1, 0, :] = 1
    patterns = _quiet(model.extract_patterns)
    np.testing.assert_array_equal(patterns['comp_inds']['row'],
                                  [n_comp - 1] * N_FOLDS)
    np.testing.assert_array_equal(patterns['comp_inds']['col'],
                                  [1] * N_FOLDS)


def test_input_covariance_is_the_row_covariance():
    """input_covariance is the covariance of the rows over (sample,
    column) observations, centred over samples: symmetric and positive
    semidefinite, with a positive trace (the diagonals of the inputs are
    zero)."""
    X, _ = make_data()
    X = X[:20]
    dcov = input_covariance(X)
    Xc = X - X.mean(0)
    for k in range(N_FREQS):
        rows = Xc[..., k].transpose(1, 0, 2).reshape(N_NODES, -1)
        expected = rows @ rows.T / (len(X) - 1)
        np.testing.assert_allclose(dcov[..., k], expected, rtol=1e-5,
                                   atol=1e-8)
        np.testing.assert_allclose(dcov[..., k], dcov[..., k].T, atol=1e-10)
        assert np.linalg.eigvalsh(dcov[..., k]).min() > -1e-8
        assert np.trace(dcov[..., k]) > 0


def test_patterns_use_each_folds_validation_data(tfr_path):
    """During CV training, each fold's covariance comes from that fold's
    own validation data (it used to be the first fold's in every fold),
    recompute_input_covariance reproduces it, and the spatial patterns are
    dcov @ w for the chosen component and frequency."""
    # Dataset's train_batch sets the validation batch (the training batch
    # comes from the metadata): make it cover a whole validation fold (as
    # in the age pipeline: batches of 50, folds of 48)
    model = make_model(tfr_path, 'symmetric', model_id='dcov_folds',
                       train_batch=32)
    assert all(len(f) <= 32 for f in model.dataset.h_params['folds'][0])
    train_cv(model)
    dcov = np.array(model.cv_patterns['dcov'])
    for fold in range(N_FOLDS):
        _, val = model.dataset._build_dataset(
            model.dataset.h_params['train_paths'],
            train_batch=model.dataset.training_batch, test_batch=None,
            split=True, val_fold_ind=fold)
        X, _ = next(iter(val))
        np.testing.assert_allclose(dcov[..., fold], input_covariance(X),
                                   rtol=1e-5, atol=1e-8)
    assert not np.allclose(dcov[..., 0], dcov[..., 1])

    model.cv_patterns['dcov'] = np.zeros_like(dcov)
    recomputed = _quiet(model.recompute_input_covariance)
    np.testing.assert_allclose(recomputed, dcov, rtol=1e-5, atol=1e-8)

    patterns = _quiet(model.extract_patterns)
    w = model.meta.weights['ssum1']
    for key in ('row', 'col', 'diag'):
        freq = patterns['freq_inds'][key]
        comp = patterns['comp_inds'][key]
        kernels = patterns['freq_patterns'][key]
        assert kernels.shape == (N_FREQS, N_FOLDS)
        best = (np.abs(kernels) if key == 'diag' else kernels).argmax(0)
        np.testing.assert_array_equal(freq, best)
        for i in range(N_FOLDS):
            np.testing.assert_allclose(
                patterns['spatial_patterns'][key][:, i],
                dcov[:, :, freq[i], i] @ w[freq[i], :, comp[i], i],
                rtol=1e-5, atol=1e-8)


def test_row_and_col_take_their_pairs_frequency(tfr_path):
    """The pointwise layer weights the pair (row, col) with the row
    component's frequency weights (see
    test_trainable_pointwise_weights_each_component), so the row and col
    components take those weights (of the SquareSum3d output) and their
    largest frequency, not the col component's own weights."""
    model = make_model(tfr_path, 'symmetric', model_id='pair_freq')
    train_cv(model)
    patterns = _quiet(model.extract_patterns)
    pointwise = model.meta.weights['pointwise_effective']
    for i in range(N_FOLDS):
        row = patterns['comp_inds']['row'][i]
        pair = pointwise[:, row, i]
        for key in ('row', 'col'):
            np.testing.assert_array_equal(
                patterns['freq_patterns'][key][:, i], pair)
            assert patterns['freq_inds'][key][i] == np.argmax(pair)


def test_patterns_need_the_effective_pointwise_weights(tfr_path):
    """Models trained before the effective pointwise weights were stored
    can't give frequency patterns when batch normalization or attention
    scales the frequencies: extract_patterns asks to retrain them."""
    model = make_model(tfr_path, 'symmetric', model_id='no_effective')
    train_cv(model)
    del model.meta.weights['pointwise_effective']
    with pytest.raises(ValueError, match='Retrain'):
        model.extract_patterns()


def test_activations_are_the_models(tfr_path):
    """compute_patterns' activations go through the model's layers (incl.
    the batch normalization and attention before the pointwise layer):
    the output activations are the model's predictions."""
    model = make_model(tfr_path, 'symmetric', model_id='activations')
    assert model.specs['attention_rate'] > 0 and model.ssum_bn is not None
    train_cv(model)
    X, _ = next(iter(model.dataset.val))
    activations = model._activations(X)
    np.testing.assert_allclose(activations['fin_fc'],
                               model.km(X, training=False), rtol=1e-5,
                               atol=1e-5)
    se_scale = activations['se_scale']
    assert se_scale.shape == (len(X), N_FREQS)
    assert np.all((se_scale > 0) & (se_scale < 1))


def test_effective_pointwise_weights_are_those_of_the_squaresum_output(
        tfr_path):
    """Without attention, the pointwise layer is linear in the SquareSum3d
    output (batch normalization in inference mode is affine): its output
    at (a, c), where active, is a fixed linear function of ssum[a, c, :]
    with the weights pointwise_effective[:, a]."""
    model = make_model(tfr_path, 'symmetric', model_id='effective',
                       attention_rate=0)
    # Batch normalization that changes the weights: scale, shift and
    # statistics differ by frequency
    rng = np.random.default_rng(1)
    bn = model.ssum_bn
    bn.gamma.assign(rng.uniform(0.5, 2, N_FREQS) * rng.choice([-1, 1],
                                                                N_FREQS))
    bn.beta.assign(rng.normal(size=N_FREQS))
    bn.moving_mean.assign(rng.uniform(0, 1, N_FREQS))
    bn.moving_variance.assign(rng.uniform(0.1, 3, N_FREQS))
    X, _ = make_data()
    activations = model._activations(X)
    ssum, out = activations['squaresym'], activations['pointwise'][..., 0]
    weights = model._effective_pointwise()
    kernel = np.squeeze(model.pointwise.kernel.numpy())
    assert not np.allclose(weights, kernel)
    n_fitted = 0
    for a in range(ssum.shape[1]):
        active = out[:, a, :] > 1e-3
        if active.sum() < 2 * N_FREQS:
            continue
        design = np.column_stack([ssum[:, a][active],
                                  np.ones(active.sum())])
        coef = np.linalg.lstsq(design, out[:, a][active], rcond=None)[0]
        np.testing.assert_allclose(coef[:-1], weights[:, a], rtol=1e-3,
                                   atol=1e-4)
        n_fitted += 1
    assert n_fitted > 0


def test_each_cv_fold_starts_with_a_fresh_optimizer(tfr_path, monkeypatch):
    """Each CV fold trains from a fresh optimizer: iteration 0, zero Adam
    moments, the initial learning rate (they used to carry over from the
    previous fold)."""
    model = make_model(tfr_path, 'symmetric', model_id='fresh_optimizer')
    opt = model.km.optimizer
    fit = model.km.fit
    starts, ends = [], []

    def state():
        return {v.path: np.array(v.numpy()) for v in opt.variables}

    def recording_fit(*args, **kwargs):
        starts.append(state())
        history = fit(*args, **kwargs)
        ends.append(state())
        return history

    monkeypatch.setattr(model.km, 'fit', recording_fit)
    train_cv(model)
    assert len(starts) == N_FOLDS
    for start, end in zip(starts, ends):
        for path, value in start.items():
            if path.endswith('learning_rate'):
                np.testing.assert_allclose(value, 1e-2)
            else:
                # iteration count and moments
                assert not np.any(value), path
        assert end[opt.iterations.path] > 0


def test_cv_training_saves_each_folds_weights(tfr_path):
    """With save_fold_weights, each fold's model can be loaded after
    training: its weights reproduce that fold's test predictions."""
    model = make_model(tfr_path, 'symmetric', model_id='fold_weights')
    _quiet(
        model.train,
        n_epochs=2,
        eval_step=3,
        early_stopping=1,
        mode='cv',
        store_test_predictions=True,
        save_fold_weights=True,
    )
    X, _ = make_data()
    X_test = X[np.asarray(model.meta.data['test_fold'])]
    for fold in range(N_FOLDS):
        path = model.fold_weights_path(fold)
        assert os.path.exists(path)
        model.km.load_weights(path)
        pred = np.ravel(model.km(X_test, training=False))
        np.testing.assert_allclose(
            pred,
            np.ravel(model.cv_test_predictions[fold]['y_pred']),
            rtol=1e-4,
            atol=1e-4,
        )


def test_feature_relevance_is_on_the_features_the_model_reads(tfr_path):
    """With the dense layer reading one SquareSum3d pair only, all the
    relevance is on that pair."""
    model = make_model(tfr_path, 'symmetric', model_id='relevance')
    n_latent = model.specs['n_latent']
    a, b = 1, 2
    kernel = np.abs(model.fc1.kernel.numpy())
    keep = np.zeros(kernel.shape[0], bool)
    keep[a * n_latent + b] = True  # flattened (a, b, pointwise)
    kernel[~keep] = 0
    model.fc1.kernel.assign(kernel)
    # A path through the ReLUs: positive weights after the features
    model.fc1.bias.assign(np.zeros_like(model.fc1.bias.numpy()))
    model.pointwise.kernel.assign(np.abs(model.pointwise.kernel.numpy()))
    model.fc.kernel.assign(np.ones_like(model.fc.kernel.numpy()))
    X, _ = make_data()
    R = model.feature_relevance(X[:32])
    assert R.shape == (n_latent, n_latent, N_FREQS)
    assert np.all(R >= 0)
    # The pointwise layer of pair (a, b) weights all frequencies of S[a, b].
    # The other features reach the output only through the attention
    # block's pooling, a small effect.
    others = np.ones_like(R, bool)
    others[a, b] = False
    assert R[a, b].sum() > 0
    assert R[others].max() < 0.01 * R[a, b].max()


def test_combined_pattern_of_a_single_relevant_filter(tfr_path, monkeypatch):
    """With all the relevance on one filter at one frequency, the combined
    pattern is that filter's |z-scored| activation pattern."""
    model = make_model(tfr_path, 'symmetric', model_id='combined_single')
    n_latent = model.specs['n_latent']
    a, f = 2, 1
    R = np.zeros((n_latent, n_latent, N_FREQS))
    R[a, a, f] = 1.
    monkeypatch.setattr(model, 'feature_relevance', lambda X: R)
    X, _ = make_data()
    dcov = input_covariance(X)
    out = model.combined_pattern(X, dcov)
    w = model.squaresum_3d.w.numpy()[f, :, a]
    p = np.abs(dcov[:, :, f] @ w)
    np.testing.assert_allclose(out['pattern'], (p - p.mean()) / p.std(),
                               rtol=1e-5, atol=1e-6)
    expected = np.zeros(N_FREQS)
    expected[f] = 1
    np.testing.assert_allclose(out['freq_relevance'], expected)
    assert out['filter_relevance'][a, f] == 1


def test_combined_patterns_use_each_folds_model_and_data(tfr_path):
    """Each fold's combined pattern comes from that fold's weights and
    validation data."""
    model = make_model(tfr_path, 'symmetric', model_id='combined_folds')
    _quiet(
        model.train,
        n_epochs=2,
        eval_step=3,
        early_stopping=1,
        mode='cv',
        collect_patterns=True,
        save_fold_weights=True,
    )
    out = _quiet(model.extract_combined_patterns)
    n_latent = model.specs['n_latent']
    assert out['pattern'].shape == (N_NODES, N_FOLDS)
    assert out['freq_relevance'].shape == (N_FREQS, N_FOLDS)
    assert out['filter_relevance'].shape == (n_latent, N_FREQS, N_FOLDS)
    np.testing.assert_allclose(out['freq_relevance'].sum(0), 1, rtol=1e-5)
    fold = 1
    model.km.load_weights(model.fold_weights_path(fold))
    X = model._validation_data(fold)
    expected = model.combined_pattern(X, input_covariance(X))
    np.testing.assert_allclose(out['pattern'][:, fold], expected['pattern'],
                               rtol=1e-4, atol=1e-5)


def test_combined_patterns_need_each_folds_weights(tfr_path):
    model = make_model(tfr_path, 'symmetric', model_id='combined_missing')
    missing = [os.path.join(model.model_path, f'missing_{k}.weights.h5')
               for k in range(N_FOLDS)]
    with pytest.raises(FileNotFoundError, match='save_fold_weights'):
        model.extract_combined_patterns(weight_paths=missing)
