#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import os
import h5py
import scipy.misc

import tensorflow as tf
import numpy as np

from copy import deepcopy
from sacred import Experiment

from auto_encoder import AutoEncoder

ex = Experiment("binding_dae")


class DAEModel(object):
    @ex.capture
    def __init__(self, is_training, network, training, optimization):
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
        self._nvars = get_variable_total(tvars, verbose=True)

        opt = None
        if optimization['name'] == 'adam':
            opt = tf.train.AdamOptimizer()
        elif optimization['name'] == 'sgd':
            opt = tf.train.GradientDescentOptimizer(self.lr)
        elif optimization['name'] == 'adagrad':
            opt = tf.train.AdagradOptimizer(self.lr)
        elif optimization['name'] == 'sgd-mom':
            opt = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True)

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
        'train_set': 'train_multi'  # {train_multi, train_single}
    }

    noise = {
        'name': 'salt_n_pepper',
        'param': {'ratio': 0.5,
                  'probability': 0.5
        }
    }

    optimization = {
        'name': 'sgd',
        'lr': 0.05,
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

    save_path = './networks'
    verbose = True


@ex.capture(prefix='dataset')
def open_dataset(path, name):
    filename = os.path.join(path, name + '.h5')

    return h5py.File(filename, 'r')


@ex.capture(prefix='dataset')
def get_raw_data(path, name, train_set):
    with open_dataset(path, name) as f:
        train_size = int(0.9 * f[train_set]['default'].shape[1])
        train_data = f[train_set]['default'][:, :train_size]
        valid_data = f[train_set]['default'][:, train_size:]
        test_data = f['test']['default'][:]

    return train_data, valid_data, test_data


@ex.capture(prefix='training')
def run_epoch(session, m, data, train_op, batch_size):
    """Runs the model on the given data."""
    costs = 0.0

    # run through the epoch
    for step, x in enumerate(iterator(data, batch_size)):
        feed_dict = {m.input_data: x, m.targets: noisify(data=x)}

        # run batch
        cost, outputs, _ = session.run([m.cost, m.outputs, train_op], feed_dict)

        # update stats
        costs += cost

    return costs/(step+1), outputs


def iterator(data, batch_size):
    """
    Iterates through the data
    :param data: inputs (assumes data to be in format (T, B, ROW, COLUMN, CH))
    :param batch_size: batch_size
    :return: yield batch at each call
    """
    epoch_size = data.shape[1] // batch_size

    for i in range(epoch_size):
        yield data[0, i*batch_size: (i+1)*batch_size, :, :].reshape(batch_size, -1)

    # yield data[0, epoch_size*batch_size:, :, :].reshape(batch_size, -1)


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


def get_variable_total(_vars, verbose=True):
    total_n_vars = 0
    for var in _vars:
        sh = var.get_shape().as_list()
        total_n_vars += np.prod(sh)

        if verbose:
            print(var.name, sh)

    if verbose:
        print(total_n_vars, 'total variables')

    return total_n_vars


def save_image(filename, image_array):
    scipy.misc.imsave(filename, image_array)


def delete_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


@ex.automain
def run(seed, save_path, network, optimization, training, _run):
    ex.commands['print_config']()

    # clear temp folder
    delete_files('temp_output')

    # seed numpy
    np.random.seed(seed)

    # load data
    train_data, valid_data, test_data = get_raw_data()
    save_img_ind = np.random.randint(test_data.shape[1])
    save_image('temp_output/test_image.jpg', test_data[0, save_img_ind, :, :, 0])

    # unpack some vars
    lr, lr_decay, lr_decay_after_epoch = optimization['lr'], optimization['lr_decay'], optimization['lr_decay_after_epoch']

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
            mvalid = DAEModel(is_training=False)

            test_training = deepcopy(training)
            test_training.update({'batch_size': test_data.shape[1]})
            mtest = DAEModel(is_training=False, training=test_training)

        # init all variables
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        trains, vals, tests, best_val = [np.inf], [np.inf], [np.inf], np.inf

        # start training
        patience = 0
        for i in range(training['max_epoch']):

            if patience == training['max_patience']:
                break

            # compute decay -> learning rate and assign in model
            lr_decay = lr_decay ** max(i - lr_decay_after_epoch, 0.0)
            m.assign_lr(session, lr * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

            # evaluate models
            train_loss, _ = run_epoch(session, m, train_data, m.train_op)
            print("Epoch: %d Train Loss: %.3f" % (i + 1, train_loss))

            valid_loss, _ = run_epoch(session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid Loss: %.3f" % (i + 1, valid_loss))

            test_loss, test_outputs = run_epoch(session, mtest, test_data, tf.no_op(), batch_size=test_training['batch_size'])
            print("Test Loss: %.3f" % test_loss)

            # update logs and save if old valid has improved
            trains.append(train_loss)
            if valid_loss < best_val:
                best_val = valid_loss
                print("Best valid Loss improved to %.03f" % best_val)
                save_destination = saver.save(session, os.path.join(save_path, str(seed) + "_best_model.ckpt"))
                print("Saved to:", save_destination)

                # save image of best epoch so far
                save_image('temp_output/recon_test_image_e{}.jpg'.format(i), np.array(test_outputs)[save_img_ind, :].reshape(28, 28))

                patience = 0
            else:
                patience += 1

            vals.append(valid_loss)
            tests.append(test_loss)

            # update logs in sacred
            _run.info['epoch_nr'] = i + 1
            _run.info['nr_parameters'] = m.nvars.item()
            _run.info['logs'] = {'train_loss': trains,
                                 'valid_loss': vals,
                                 'test_loss': tests}

        print("Training is over.")
        best_val_epoch = np.argmin(vals)
        print("Best validation loss %.03f was at Epoch %d" % (vals[best_val_epoch], best_val_epoch))
        print("Training loss at this Epoch was %.03f" % trains[best_val_epoch])
        print("Test loss at this Epoch was %.03f" % tests[best_val_epoch])
        _run.info['best_val_epoch'] = best_val_epoch
        _run.info['best_valid_loss'] = vals[best_val_epoch]

    return vals[best_val_epoch]