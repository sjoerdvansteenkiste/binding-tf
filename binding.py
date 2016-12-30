#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import os
import h5py
import utils
import time
import pickle

import tensorflow as tf
import numpy as np
import reconstruction_clustering as rc

from sacred import Experiment
from auto_encoder import AutoEncoder
from reconstruction_clustering import ReconstructionClustering


ex = Experiment("binding")


class BindingModel(object):
    @ex.capture
    def __init__(self, is_training, network, em, dataset):
        _shape = dataset['shape']

        # AE vars
        _input_size = _shape[0] * _shape[1] * _shape[2]
        self._is_training = is_training
        self.ae_input_data = tf.placeholder(tf.float32, [None, _input_size])
        self.targets = tf.placeholder(tf.float32, [None, _input_size])

        # model
        self._ae = AutoEncoder(network['hidden_size'], _input_size, network['e_activation'],
                               network['d_activation'], tied=network['tied'])

        self.outputs, self.cost, self.lr, self.n_vars, self.train_op = 5*[None]
        self.build_network_training_model(dataset['binary'])

        # RC vars
        self.rc_input_data = tf.placeholder(tf.float32, [1, None] + _shape + [1])
        self.pi = tf.placeholder(tf.float32, [1, None, 1, 1, 1, em['k']])
        self.gamma = tf.placeholder(tf.float32, [1, None] + _shape[:2] + [1, em['k']])

        # RC model
        _distribution = 'binomial' if dataset['binary'] else 'gaussian'
        self._rc = ReconstructionClustering(self._ae, _shape, em['k'], em['e_step'], _distribution, dataset['binary'])

        self.new_gamma, self.new_pi, self.likelihood_post_m, self.likelihood_post_e = 4*[None]
        self.build_reconstruction_clustering_model()

    def build_network_training_model(self, binary):
        self.outputs, state_dict = self._ae(self.ae_input_data)

        if binary:
            pixel_loss = tf.reduce_sum(self.binomial_cross_entropy_loss(self.outputs, self.targets), 1)
        else:
            pixel_loss = tf.reduce_sum(self.squared_error_loss(self.outputs, self.targets), 1)

        loss = tf.reduce_mean(pixel_loss)
        self.cost = loss

        if not self._is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)

        # print total number of trainable variables
        tvars = tf.trainable_variables()
        self.n_vars = utils.get_variable_total(tvars, verbose=True)

        # create optimizer
        opt = tf.train.GradientDescentOptimizer(self.lr)

        self.train_op = opt.minimize(loss)

    def build_reconstruction_clustering_model(self, ):
        self.new_gamma, self.new_pi, self.likelihood_post_m, self.likelihood_post_e = \
            self._rc(self.rc_input_data, self.gamma, self.pi)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @staticmethod
    def binomial_cross_entropy_loss(y, t):
        bce = t * tf.log(tf.clip_by_value(y, 1e-6, 1.)) + (1. - t) * tf.log(tf.clip_by_value(1 - y, 1e-6, 1.))

        return -bce

    @staticmethod
    def squared_error_loss(y, t):
        se = (y - t)**2
        return se


@ex.config
def cfg():
    dataset = {
        'name': 'shapes',
        'path': './data',
        'train_set': 'train_single',  # {train_multi, train_single}
        'split': 0.9,
        'shape': (28, 28, 1),
        'binary': True
    }

    noise = {
        'name': 'salt_n_pepper',
        'probability': 0.5,
        'dynamic': False,               # if dynamic set to true it will ignore the params
        'param': {'ratio': 0.5,
        }
    }

    optimization = {
        'lr': 0.1,
        'lr_decay': 0.98,
        'lr_decay_after_epoch': 10
    }

    training = {
        'max_patience': 10,
        'batch_size': 100,
        'max_epoch': 500
    }

    network = {
        'name': 'AE',
        'hidden_size': [500],              # hidden sizes of the encoder - decoder.
        'init_scale': 0.1,
        'e_activation': ['tanh'],               # encoder activations
        'd_activation': ['sigmoid'],            # decoder activations
        'tied': False,
    }

    em = {
        'nr_iters': 10,
        'k': 3,
        'nr_samples': np.inf,
        'e_step': 'expectation',  # {expectation, expectation_pi, max, or max_pi}
        'init_type': 'gaussian',  # {gaussian, uniform, or spatial}
        'suffix': None            # specify the suffix, None is default, use None to not save
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
    data = {}
    with open_dataset(path, name) as f:
        if name in ['bars', 'corners', 'easy_superpos', 'mnist_shapes', 'multi_mnist_thresholded', 'shapes']:
            train_size = int(split * f[train_set]['default'].shape[1])
            data['train_data'] = f[train_set]['default'][:, :train_size]
            data['train_groups'] = f[train_set]['groups'][:, :train_size]
            data['valid_data'] = f[train_set]['default'][:, train_size:]
            data['valid_groups'] = f[train_set]['groups'][:, train_size:]
        else:
            raise ValueError('Unknown dataset "{}"'.format(name))

    return data


@ex.capture(prefix='dataset')
def get_test_data(path, name):
    data = {}
    with open_dataset(path, name) as f:
        if name in ['bars', 'corners', 'easy_superpos', 'mnist_shapes', 'multi_mnist_thresholded', 'shapes']:
            data['test_data'] = f['test']['default'][:]
            data['test_groups'] = f['test']['groups'][:]
        else:
            raise ValueError('Unknown dataset "{}"'.format(name))

    return data


@ex.capture()
def run_epoch(session, m, data, targets, train_op, training, network):
    """Runs the training part on the given data."""
    step = 0
    costs = 0.0
    total_outputs = []

    # run through the epoch
    for step, (x, y) in enumerate(iterator(data, targets, training['batch_size'], network['name'] == 'CAE')):
        feed_dict = {m.ae_input_data: x, m.targets: y}

        # run batch
        cost, outputs, _ = session.run([m.cost, m.outputs, train_op], feed_dict)

        # update stats
        costs += cost
        total_outputs.append(outputs)

    return costs/(step+1), np.concatenate(total_outputs)


def _run_rc_iteration(session, m, input_data, gamma, pi):
    feed_dict = {m.rc_input_data: input_data, m.gamma: gamma, m.pi: pi}

    return session.run([m.new_gamma, m.new_pi, m.likelihood_post_m, m.likelihood_post_e], feed_dict)


@ex.capture(prefix='em')
def _run_reconstruction_clustering(session, m, input_data, true_groups, k,
                                   nr_iters, init_type, binary, e_step, rnd):
        # init distribution
        distribution = 'binomial' if binary else 'gaussian'

        # obtain input dimensions and add cluster dimension
        T, N, H, W, C = input_data.shape
        input_data = input_data[..., None]  # add a cluster dimension
        true_groups = true_groups.reshape(N, -1) if true_groups is not None else None  # reshape for computing scores

        # allocate storage space
        gammas = np.zeros((nr_iters + 1, T, N, H, W, 1, k))
        likelihoods = np.zeros((2 * nr_iters + 1, N))
        scores = -1 * np.ones((nr_iters + 1, N, 2))

        # init pi and output_prior
        pi = np.ones((1, N, 1, 1, 1, k)) / k
        output_prior = np.ones_like(input_data) * 0.5

        # set initial values
        gammas[0] = rc.get_initial_groups(k=k, dims=(N, H, W), init_type=init_type, rnd=rnd, hard_assign=e_step.startswith('max'))
        likelihoods[0] = rc.get_likelihood(output_prior, input_data, gammas[0], distribution)

        if true_groups is not None:
            scores[0, :, 0], scores[0, :, 1] = rc.evaluate_groups(true_groups, gammas[0].reshape(N, -1, k))

        # run rc for specified number of iterations
        for j in range(nr_iters):
            gammas[j + 1], pi, likelihoods[2 * j + 1],  likelihoods[2 * j + 2] = _run_rc_iteration(
                session, m, input_data, gammas[j], pi)

            # save the score and confidence
            if true_groups is not None:
                scores[j + 1, :, 0], scores[j + 1, :, 1] = rc.evaluate_groups(true_groups, gammas[j + 1].reshape(N, -1, k))

        return gammas, likelihoods, scores


def iterator(data, targets, batch_size, use_convolution):
    epoch_size = data.shape[1] // batch_size  # TODO enable for the remainder

    for i in range(epoch_size):
        if use_convolution:
            yield (data[0, i * batch_size: (i + 1) * batch_size], targets[0, i * batch_size: (i + 1) * batch_size])
        else:
            yield (data[0, i*batch_size: (i+1)*batch_size].reshape(batch_size, -1),
                   targets[0, i*batch_size: (i+1)*batch_size].reshape(batch_size, -1))

    # yield (data[0, epoch_size*batch_size:, :, :].reshape(batch_size, -1),
    #        targets[0, epoch_size*batch_size:, :, :].reshape(batch_size, -1))


@ex.capture
def train_dae(seed, network, optimization, training, dataset, data, net_folder_path, debug, _run, **kwargs):
    # unpack some vars
    lr, base_lr_decay, lr_dea = optimization['lr'], optimization['lr_decay'], optimization['lr_decay_after_epoch']
    net_file_path = os.path.join(net_folder_path, "best_model.ckpt")

    with tf.Graph().as_default(), tf.Session() as session:

        # seed tensorflow
        tf.set_random_seed(seed)

        # define initializer
        initializer = tf.random_uniform_initializer(-network['init_scale'], network['init_scale'])

        # init train DAE model
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = BindingModel(is_training=True)

        # init valid DAE models re-using the variables used at training time
        with tf.variable_scope("model", reuse=True, initializer=initializer):        # TODO currently redundant
            valid_m = BindingModel(is_training=False)

        # init all variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        train_losses, valid_losses, train_scores, valid_scores, best_val = [np.inf], [np.inf], [-np.inf], [-np.inf], np.inf

        # start training
        patience = 0
        for i in range(training['max_epoch']):

            if patience == training['max_patience']:
                break

            # compute decay -> learning rate and assign in model
            lr_decay = base_lr_decay ** max(i - lr_dea, 0.0)
            m.assign_lr(session, lr * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

            # produce data
            t = time.time()
            train_data, train_targets = noisify(data=data['train_data']), data['train_data']

            # evaluate and update training model
            train_loss, _ = run_epoch(session, m, train_data, train_targets, m.train_op)
            print("Epoch: %d Train Loss: %.3f, took %.3fs" % (i + 1, train_loss, time.time()-t))

            # produce data
            t = time.time()
            valid_data, valid_targets = noisify(data=data['valid_data']), data['valid_data']

            # evaluate valid model
            valid_loss, valid_outputs = run_epoch(session, valid_m, valid_data, valid_targets, tf.no_op())
            print("Epoch: %d Valid Loss: %.3f, took %.3fs" % (i + 1, valid_loss, time.time()-t))

            # save if old valid has improved
            if valid_loss < best_val:
                best_val = valid_loss
                print("Best valid Loss improved to %.03f" % best_val)
                save_destination = saver.save(session, net_file_path)
                print("Saved to:", save_destination)

                # save image of best epoch so far
                if debug:
                    input_valid_image = valid_data[0, kwargs['save_img_ind']]
                    utils.save_image('debug_output/input_valid_image_e{}.jpg'.format(i), input_valid_image)

                    output_valid_image = valid_outputs[kwargs['save_img_ind']].reshape(dataset['shape'])
                    utils.save_image('debug_output/recon_valid_image_e{}.jpg'.format(i), output_valid_image)

                patience = 0
            else:
                patience += 1

            # update logs of losses and scores
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # update logs in sacred
            _run.info['epoch_nr'] = i + 1
            _run.info['nr_parameters'] = m.n_vars.item()
            _run.info['logs'] = {'train_loss': train_losses, 'valid_loss': valid_losses}

        # add network to db
        ex.add_artifact(net_file_path)

        # log results
        print("Training is over.")
        best_val_epoch = np.argmin(valid_losses)
        print("Best validation loss %.03f was at Epoch %d" % (valid_losses[best_val_epoch], best_val_epoch))
        print("Training loss at this Epoch was %.03f" % train_losses[best_val_epoch])
        _run.info['best_val_epoch'] = best_val_epoch
        _run.info['best_valid_loss'] = valid_losses[best_val_epoch]


@ex.capture(prefix='noise')
def noisify(name, probability, dynamic, param, data):
    if name == 'salt_n_pepper':
        ratio = np.sum(data) / np.prod(data.shape) if dynamic else param['ratio']
        noisy_data = salt_and_pepper_noise(data, probability, ratio)
    elif name == 'gaussian':
        mu, sigma = (np.mean(data), np.std(data)) if dynamic else (param['mu'], param['sigma'])
        noisy_data = gaussian_noise(data, probability, mu, sigma)
    else:
        raise ValueError('Unknown noise "{}"'.format(name))

    return noisy_data


def salt_and_pepper_noise(data, probability, ratio):
    noisy_data = data.copy()

    r = np.random.rand(*noisy_data.shape)
    noisy_data[r >= 1.0 - probability * ratio] = 1.0  # salt
    noisy_data[r <= probability * (1.0 - ratio)] = 0.0  # pepper

    return noisy_data


def gaussian_noise(data, probability, mu, sigma):
    noisy_data = data.copy()

    r = np.random.rand(*noisy_data.shape)
    n = sigma * np.random.randn(*noisy_data.shape) + mu
    idx = r >= 1.0 - probability
    noisy_data[idx] = n[idx]

    return noisy_data


@ex.capture
def perform_reconstruction_clustering(session, m, em, data, groups, binary, rnd, em_dump_path=None):
    nr_samples = min(em['nr_samples'], data.shape[1])
    sliced_data = data[:, :nr_samples]
    sliced_groups = groups[:, :nr_samples] if groups is not None else None

    all_gammas, all_likelihoods, all_scores = _run_reconstruction_clustering(
        session, m, sliced_data, sliced_groups, binary=binary, rnd=rnd)

    all_gammas = np.swapaxes(all_gammas, 1, 2)
    all_gammas = np.swapaxes(all_gammas, 0, 1)
    all_likelihoods = np.swapaxes(all_likelihoods, 0, 1)
    all_scores = np.swapaxes(all_scores, 0, 1)

    print('Average Score: {:.4f}'.format(all_scores[:, -1, 0].mean()))
    print('Average Confidence: {:.4f}'.format(all_scores[:, -1, 1].mean()))

    if em_dump_path is not None:
        with open(em_dump_path, 'wb') as f:
            pickle.dump((all_scores, all_likelihoods, all_gammas), f)
        print('wrote the results to {}'.format(em_dump_path))
    return all_scores[:, -1, 0].mean()


def perform_reconstruction_clustering_from_file(net_folder_path, data, groups, rnd, em=None, em_dump_path=None):
    """ Support external usage, i.e. loading a trained network and rebinding it elsewhere"""
    net_filename_path = os.path.join(net_folder_path, "best_model.ckpt")
    model_config_path = os.path.join(net_folder_path, "model_config.pickle")

    # load config
    with open(model_config_path, 'rb') as f:
        model_config = pickle.load(f)

    network, dataset = model_config['network'], model_config['dataset']
    em = em if em is not None else model_config['em']

    # create new session
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model"):
            m = BindingModel(False, network, em, dataset)

        # restore params
        saver = tf.train.Saver()
        saver.restore(session, net_filename_path)

        # run reconstruction clustering for dea
        return perform_reconstruction_clustering(session=session, m=m, em=em, data=data, groups=groups,
                                                 binary=dataset['binary'], rnd=rnd, em_dump_path=em_dump_path)


@ex.automain
def run(seed, save_path, network, em, dataset, _rnd, debug):
    ex.commands['print_config']()

    # create storage directories if they don't exist yet
    utils.create_directory('networks')
    utils.create_directory('results')

    if debug:
        utils.create_directory('debug_output')
        utils.delete_files('debug_output')

    # seed numpy
    np.random.seed(seed)

    # load data
    data = get_training_data()

    # if debug set to true the trainer will report on the dae performance
    save_img_ind = np.random.randint(data['valid_data'].shape[1])
    if debug:
        utils.save_image('debug_output/valid_image.jpg', data['valid_data'][0, save_img_ind])

    # produce saving directory filenames
    folder_name = dataset['name'] + "_" + dataset['train_set'] + "_" + time.strftime(
        "%d_%m_%Y_%H:%M:%S") + "_" + str(seed)
    net_folder_path = os.path.join(save_path, folder_name)
    utils.create_directory(net_folder_path)

    model_config_path = os.path.join(net_folder_path, "model_config.pickle")
    em_dump_path = None

    if em['suffix'] is not None:
        em_dump_path = 'results/{}_{}_{}{}.pickle'.format(dataset['name'], em['nr_iters'], em['k'], em['suffix'])

    # save net structure
    with open(model_config_path, 'wb') as f:
        pickle.dump({'network': network, 'em': em, 'dataset': dataset}, f)

    # train dea
    train_dae(data=data, net_folder_path=net_folder_path, **{'save_img_ind': save_img_ind})

    # load test data
    data = get_test_data()
    test_data = data['test_data']
    test_groups = data['test_groups'] if 'test_groups' in data.keys() else None

    # perform rc on best model
    return perform_reconstruction_clustering_from_file(net_folder_path, test_data, test_groups, _rnd, em_dump_path=em_dump_path)
