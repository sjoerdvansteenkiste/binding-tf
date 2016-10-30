#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import os
import h5py
import utils

import tensorflow as tf
import numpy as np

from sacred import Experiment
from auto_encoder import AutoEncoder
from sklearn.metrics import adjusted_mutual_info_score

ex = Experiment("binding_dae")


class DAEModel(object):
    @ex.capture
    def __init__(self, is_training, network, training):
        self._is_training = is_training

        self._input_data = tf.placeholder(tf.float32, [training['batch_size'], network['input_size']])
        self._targets = tf.placeholder(tf.float32, [training['batch_size'], network['input_size']])

        if network['hidden_activation'] == 'tanh':
            ae = AutoEncoder(network['hidden_size'], network['input_size'],
                             hidden_activation=tf.tanh, tied=network['tied'])
        elif network['hidden_activation'] == 'relu':
            ae = AutoEncoder(network['hidden_size'], network['input_size'],
                             hidden_activation=tf.nn.relu, tied=network['tied'])
        else:
            ae = AutoEncoder(network['hidden_size'], network['input_size'], tied=network['tied'])

        self._outputs, state_dict = ae(self._input_data)

        pixel_ce = binomial_cross_entropy_loss(self._outputs, self._targets)
        loss = tf.reduce_mean(tf.reduce_sum(pixel_ce, reduction_indices=1))
        self._cost = loss

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)

        # print total number of trainable variables
        tvars = tf.trainable_variables()
        self._nvars = utils.get_variable_total(tvars, verbose=True)

        # create optimizer
        opt = tf.train.GradientDescentOptimizer(self.lr)

        self._train_op = opt.minimize(loss)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def outputs(self):
        return self._outputs

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def nvars(self):
        return self._nvars


@ex.config
def cfg():
    dataset = {
        'name': 'shapes',
        'path': './data',
        'train_set': 'train_single',  # {train_multi, train_single}
        'split': 0.9
    }

    noise = {
        'name': 'salt_n_pepper',
        'param': {'ratio': 0.5,
                  'probability': 0.5
        }
    }

    optimization = {
        'lr': 0.01,
        'lr_decay': 1.0,
        'lr_decay_after_epoch': 500
    }

    training = {
        'max_patience': 10,
        'batch_size': 100,
        'max_epoch': 500
    }

    network = {
        'input_size': 28*28,
        'hidden_size': 250,
        'init_scale': 0.1,
        'hidden_activation': 'tanh',
        'tied': False
    }

    em = {
        'nr_iters': 10,
        'k': 3,
        'nr_samples': 1000,
        'e_step': 'expectation',  # {expectation, expectation_pi, max, or max_pi}
        'init_type': 'gaussian',  # {gaussian, uniform, or spatial}
        'dump_results': None
    }

    save_path = './networks'
    verbose = True
    debug = False


@ex.capture(prefix='dataset')
def open_dataset(path, name):
    filename = os.path.join(path, name + '.h5')

    return h5py.File(filename, 'r')


@ex.capture(prefix='dataset')
def get_training_data(path, name, train_set, split):
    with open_dataset(path, name) as f:
        train_size = int(split * f[train_set]['default'].shape[1])
        train_data = f[train_set]['default'][:, :train_size]
        valid_data = f[train_set]['default'][:, train_size:]

    return train_data, valid_data


@ex.capture(prefix='dataset')
def get_test_data(path, name):
    with open_dataset(path, name) as f:
        test_data = f['test']['default'][:]
        test_groups = f['test']['groups'][:]

    return test_data, test_groups


@ex.capture(prefix='training')
def run_epoch(session, m, data, train_op, batch_size, targets=None):
    """Runs the model on the given data."""
    step = 0
    costs = 0.0
    total_outputs = []

    # data equals targets if not provided - input becomes noisy
    targets = targets if targets is not None else data
    data = data if targets is not None else noisify(data=data)

    # run through the epoch
    for step, (x, y) in enumerate(iterator(data, targets, batch_size)):
        feed_dict = {m.input_data: x, m.targets: y}

        # run batch
        cost, outputs, _ = session.run([m.cost, m.outputs, train_op], feed_dict)

        # update stats
        costs += cost
        total_outputs.append(outputs)

    return costs/(step+1), np.concatenate(total_outputs)


def iterator(data, targets, batch_size):
    """
    Iterates through the data
    :param data: inputs (assumes data to be in format (T, B, ROW, COLUMN, CH))
    :param targets: optional targets
    :param batch_size: batch_size
    :return: yield batch at each call
    """
    epoch_size = data.shape[1] // batch_size

    for i in range(epoch_size):
        yield (data[0, i*batch_size: (i+1)*batch_size, :, :].reshape(batch_size, -1),
               targets[0, i*batch_size: (i+1)*batch_size, :, :].reshape(batch_size, -1))

    # yield data[0, epoch_size*batch_size:, :, :].reshape(batch_size, -1)


@ex.capture
def train_dae(seed, network, optimization, training, data, net_filename, debug, _run, **kwargs):
    # unpack some vars
    lr, lr_decay, lr_dea = optimization['lr'], optimization['lr_decay'], optimization['lr_decay_after_epoch']

    with tf.Graph().as_default(), tf.Session() as session:

        # seed tensorflow
        tf.set_random_seed(seed)

        # define initializer
        initializer = tf.random_uniform_initializer(-network['init_scale'], network['init_scale'])

        # init train DAE model
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = DAEModel(is_training=True)

        # init valid DAE models re-using the variables used at training time
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_m = DAEModel(is_training=False)

        # init all variables
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        trains, vals, best_val = [np.inf], [np.inf], np.inf

        # start training
        patience = 0
        for i in range(training['max_epoch']):

            if patience == training['max_patience']:
                break

            # compute decay -> learning rate and assign in model
            lr_decay = lr_decay ** max(i - lr_dea, 0.0)
            m.assign_lr(session, lr * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

            # evaluate models
            train_loss, _ = run_epoch(session, m, data['train_data'], m.train_op)
            print("Epoch: %d Train Loss: %.3f" % (i + 1, train_loss))

            valid_loss, valid_outputs = run_epoch(session, valid_m, data['valid_data'], tf.no_op())
            print("Epoch: %d Valid Loss: %.3f" % (i + 1, valid_loss))

            # update logs and save if old valid has improved
            trains.append(train_loss)
            if valid_loss < best_val:
                best_val = valid_loss
                print("Best valid Loss improved to %.03f" % best_val)
                save_destination = saver.save(session, net_filename)
                print("Saved to:", save_destination)

                # save image of best epoch so far
                if debug:
                    utils.save_image('debug_output/recon_valid_image_e{}.jpg'.format(i),
                                     np.array(valid_outputs)[kwargs['save_img_ind'], :].reshape(28, 28))

                patience = 0
            else:
                patience += 1

            vals.append(valid_loss)

            # update logs in sacred
            _run.info['epoch_nr'] = i + 1
            _run.info['nr_parameters'] = m.nvars.item()
            _run.info['logs'] = {'train_loss': trains, 'valid_loss': vals}

        # add network to db
        ex.add_artifact(net_filename)

        # log results
        print("Training is over.")
        best_val_epoch = np.argmin(vals)
        print("Best validation loss %.03f was at Epoch %d" % (vals[best_val_epoch], best_val_epoch))
        print("Training loss at this Epoch was %.03f" % trains[best_val_epoch])
        _run.info['best_val_epoch'] = best_val_epoch
        _run.info['best_valid_loss'] = vals[best_val_epoch]


@ex.capture(prefix='noise')
def noisify(name, param, data):
    noisy_data = data.copy()
    if name == 'salt_n_pepper':
        r = np.random.rand(*noisy_data.shape)
        noisy_data[r >= 1.0 - param['probability'] * param['ratio']] = 1.0  # salt
        noisy_data[r <= param['probability'] * (1.0 - param['ratio'])] = 0.0  # pepper

    return noisy_data


def binomial_cross_entropy_loss(y, t):
    # - t * ln(y) - (1-t) * ln(1-y)
    bce = t * tf.log(tf.clip_by_value(y, 1e-6, 1.)) + (1. - t) * tf.log(tf.clip_by_value(1-y, 1e-6, 1.))

    return -bce


def load_session(net_filename):
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, net_filename)

    return session


def get_likelihood(Y, T, group_channels):
    log_loss = T * np.log(Y.clip(1e-6, 1 - 1e-6)) + (1 - T) * np.log((1 - Y).clip(1e-6, 1 - 1e-6))

    return np.sum(log_loss * group_channels)


@ex.capture(prefix='em')
def get_initial_groups(k, dims, init_type, _rnd, low=.25, high=.75):
    shape = (1, 1, dims[0], dims[1], 1, k)  # (T, B, H, W, C, K)
    if init_type == 'spatial':
        assert k == 3
        group_channels = np.zeros((dims[0], dims[1], 3))
        group_channels[:, :, 0] = np.linspace(0, 0.5, dims[0])[:, None]
        group_channels[:, :, 1] = np.linspace(0, 0.5, dims[1])[None, :]
        group_channels[:, :, 2] = 1.0 - group_channels.sum(2)
        group_channels = group_channels.reshape(shape)
    elif init_type == 'gaussian':
        group_channels = np.abs(_rnd.randn(*shape))
        group_channels /= group_channels.sum(5)[..., None]
    elif init_type == 'uniform':
        group_channels = _rnd.uniform(low, high, size=shape)
        group_channels /= group_channels.sum(5)[..., None]
    else:
        raise ValueError('Unknown init_type "{}"'.format(init_type))
    return group_channels


def evaluate_groups(true_groups, predicted):
    idxs = np.where(true_groups != 0.0)
    score = adjusted_mutual_info_score(true_groups[idxs],
                                       predicted.argmax(1)[idxs])
    confidence = np.mean(predicted.max(1)[idxs])
    return score, confidence


@ex.capture(prefix='em')
def perform_e_step(T, Y, mixing_factors, e_step, k):
    loss = (T * Y + (1 - T) * (1 - Y)) * mixing_factors
    if e_step == 'expectation':
        group_channels = loss / loss.sum(5)[..., None]
    elif e_step == 'expectation_pi':
        group_channels = loss / loss.sum(5)[..., None]
        mixing_factors = group_channels.reshape(-1, k).sum(0)
        mixing_factors /= mixing_factors.sum()
    elif e_step == 'max':
        group_channels = (loss == loss.max(5)[..., None]).astype(np.float)
    elif e_step == 'max_pi':
        group_channels = (loss == loss.max(5)[..., None]).astype(np.float)
        mixing_factors = group_channels.reshape(-1, k).sum(0)
        mixing_factors /= mixing_factors.sum()
    else:
        raise ValueError('Unknown e_type: "{}"'.format(e_step))

    return group_channels, mixing_factors


@ex.command(prefix='em')
def reconstruction_clustering(session, model, input_data, true_groups, k, nr_iters):
    T, N, H, W, C = input_data.shape
    input_data = input_data[..., None]  # add a cluster dimension

    mixing_factors = np.ones((1, 1, 1, 1, k)) / k
    gamma = get_initial_groups(dims=(H, W))
    output_prior = np.ones_like(input_data) * 0.5

    gammas = np.zeros((nr_iters + 1, 1, H, W, C, k))
    likelihoods = np.zeros(2 * nr_iters + 1)
    scores = np.zeros((nr_iters + 1, 2))

    gammas[0:1] = gamma
    likelihoods[0] = get_likelihood(output_prior, input_data, gamma)
    scores[0] = evaluate_groups(true_groups.flatten(), gamma.reshape(-1, k))

    for j in range(nr_iters):
        X = gamma * input_data
        Y = np.zeros_like(X)

        # run the k copies of the autoencoder
        for _k in range(k):
            loss, outputs = run_epoch(session, model, X[..., _k], tf.no_op(), targets=input_data[..., 0], batch_size=1)
            Y[..., _k] = outputs.reshape((1, 1, H, W, C))

        # save the log-likelihood after the M-step
        likelihoods[2*j+1] = get_likelihood(Y, input_data, gamma)
        # perform an E-step
        gamma, mixing_factors = perform_e_step(input_data, Y, mixing_factors)
        # save the log-likelihood after the E-step
        likelihoods[2*j+2] = get_likelihood(Y, input_data, gamma)
        # save the resulting group-assignments
        gammas[j+1] = gamma[0]
        # save the score and confidence
        scores[j+1] = evaluate_groups(true_groups.flatten(), gamma.reshape(-1, k))
    return gammas, likelihoods, scores


@ex.command(prefix='em')
def evaluate(session, model, nr_samples, test_data, test_groups, dump_results=None):

    all_scores = []
    all_likelihoods = []
    all_gammas = []
    nr_samples = min(nr_samples, test_data.shape[1])
    for i in range(nr_samples):
        gammas, likelihoods, scores = reconstruction_clustering(session, model, test_data[:, i:i+1], test_groups[:, i:i+1])
        all_gammas.append(gammas)
        all_likelihoods.append(likelihoods)
        all_scores.append(scores)

    all_gammas = np.array(all_gammas)
    all_likelihoods = np.array(all_likelihoods)
    all_scores = np.array(all_scores)

    print('Average Score: {:.4f}'.format(all_scores[:, -1, 0].mean()))
    print('Average Confidence: {:.4f}'.format(all_scores[:, -1, 1].mean()))

    if dump_results is not None:
        import pickle
        with open(dump_results, 'wb') as f:
            pickle.dump((all_scores, all_likelihoods, all_gammas), f)
        print('wrote the results to {}'.format(dump_results))
    return all_scores[:, -1, 0].mean()


@ex.automain
def run(seed, save_path, dataset, training, debug, _run):
    ex.commands['print_config']()

    # create storage directories if they don't exist yet
    utils.create_directory('networks')

    if debug:
        utils.create_directory('debug_output')
        utils.delete_files('debug_output')

    # seed numpy
    np.random.seed(seed)

    # load data
    train_data, valid_data = get_training_data()

    # if debug set to true the trainer will report on the dae performance
    save_img_ind = np.random.randint(valid_data.shape[1])
    if debug:
        utils.save_image('debug_output/valid_image.jpg', valid_data[0, save_img_ind, :, :, 0])

    # get filename
    net_filename = os.path.join(save_path, dataset['name'] + "_" + str(seed) + "_best_model.ckpt")

    # train dea
    data = {'train_data': train_data, 'valid_data': valid_data}
    train_dae(data=data, net_filename=net_filename, **{'save_img_ind': save_img_ind})

    # load test data
    test_data, test_groups = get_test_data()

    # create new session
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model"):
            batch_size = 1
            b_training = training.copy()
            b_training['batch_size'] = batch_size
            m = DAEModel(is_training=True, training=b_training)

        # restore params
        saver = tf.train.Saver()
        saver.restore(session, net_filename)

        # run reconstruction clustering for dea
        return evaluate(session=session, model=m, test_data=test_data, test_groups=test_groups)