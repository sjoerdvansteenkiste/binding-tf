#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import numpy as np
import tensorflow as tf
import math

from sklearn.metrics import adjusted_mutual_info_score


class ReconstructionClustering(object):
    """
    Notation is in line with EM-like models as in Bishop.
    """

    def __init__(self, ae, data_shape, k, e_step, distribution, binary):
        self._ae = ae
        self._data_shape = data_shape
        self._k = k
        self._e_step = e_step
        self._distribution = distribution
        self._binary = binary

    def __call__(self, x, gamma, pi):
        """
        Computes a single reconstruction clustering step, consisting of the following steps:
         1. (M-step) computing the new distribution parameters for each pixel mu
         2. (likelihood estimation) estimating the likelihood after the M-step
         3. (E-step) computing the new gamma and pi by maximizing the likelihood
         4. (likelihood estimation) estimating the likelihood after the E-step

        :param x: a tensor of shape (T, N, R, C, C, 1) containing the data points
        :param gamma: a tensor of shape (T, N, R, C, C, k) containing the posterior probabilities
        :param pi: a tensor of shape (1, N, 1, 1, 1, k) containing the mixing factors
        :return: new estimates of gamma, pi, and both likelihoods as computed in step 2, 4.
        """

        # run the k copies of the auto-encoder
        mu = []
        for _k in range(self._k):
            x_gamma = gamma[:, :, :, :, :, _k] * x[:, :, :, :, :, 0]

            # compute mu
            output, _ = self._ae(tf.reshape(x_gamma, [-1, tf.reduce_prod(x.get_shape()[2:])]), reuse=True)
            mu.append(tf.reshape(output, [1, -1] + self._data_shape + [1]))

        mu = tf.concat(concat_dim=5, values=mu)

        # compute the log-likelihood after the M-step
        likelihood_post_m = self.get_likelihood(mu, x, gamma)

        # perform an E-step
        new_gamma, new_pi = self.perform_e_step(mu, x, pi)

        # compute the log-likelihood after the E-step
        likelihood_post_e = self.get_likelihood(mu, x, new_gamma)

        return new_gamma, new_pi, likelihood_post_m, likelihood_post_e

    def get_likelihood(self, prediction, targets, gamma):  # TODO is + log pi missing in computing the log loss?
        if self._distribution == 'binomial':
            log_loss = targets * tf.log(tf.clip_by_value(prediction, 1e-6, 1 - 1e-6)) + (1 - targets) * tf.log(
                tf.clip_by_value(1 - prediction, 1e-6, 1 - 1e-6))
        elif self._distribution == 'gaussian':
            sigma = 0.1  # todo make dynamic
            loss = (1 / tf.sqrt((2 * math.pi * sigma ** 2))) * tf.exp(-(targets - prediction) ** 2 / (2 * sigma ** 2))
            log_loss = tf.log(tf.clip_by_value(loss, 1e-6, 1 - 1e-6))
        else:
            raise ValueError('Unknown distribution_type: "{}"'.format(self._distribution))

        return tf.reduce_sum(log_loss * gamma, [0, 2, 3, 4, 5])

    def perform_e_step(self, mu, x, pi):
        """
        Performs the E-step, in which we use the current values for the parameters to evaluate the
        posterior probabilities, or responsibilities gamma. Optionally we also adjust the mixture values

        :param mu: a tensor containing the current parameters of shape (T, N, R, C, C, k)
        :param x: a tensor of shape (T, N, R, C, C, 1) containing the data points
        :param pi: a tensor of shape (1, N, 1, 1, 1, k) containing the mixing factors
        :return: A new estimation of the posterior probabilities and optionally mixture values
        """

        # compute the loss
        if self._distribution == 'binomial':
            loss = (x * mu + (1 - x) * (1 - mu)) * pi
        elif self._distribution == 'gaussian':
            sigma = 0.1  # TODO make dynamic
            loss = (1 / tf.sqrt((2 * math.pi * sigma ** 2))) * tf.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) * pi
        else:
            raise ValueError('Unknown distribution_type: "{}"'.format(self._distribution))

        # compute the new gamma (and pi) accordingly
        if self._e_step == 'expectation':
            gamma = tf.reduce_sum(loss, 4, keep_dims=True) / tf.reduce_sum(loss, [4, 5], keep_dims=True)

        elif self._e_step == 'expectation_pi':
            gamma = tf.reduce_sum(loss, 4, keep_dims=True) / tf.reduce_sum(loss, [4, 5], keep_dims=True)
            pi = tf.reduce_sum(tf.reshape(gamma, (x.shape[1], -1, self._k)), reduction_indices=1)
            pi /= tf.reduce_sum(pi, reduction_indices=1, keep_dims=True)

        elif self._e_step == 'max':
            loss = tf.reduce_sum(loss, reduction_indices=4, keep_dims=True)
            max_loss = tf.reduce_max(loss, reduction_indices=5, keep_dims=True)
            gamma = tf.ones(tf.shape(loss)) - tf.to_float(loss < max_loss) * tf.ones(tf.shape(loss))

        elif self._e_step == 'max_pi':
            loss = tf.reduce_sum(loss, reduction_indices=4, keep_dims=True)
            max_loss = tf.reduce_max(loss, reduction_indices=5, keep_dims=True)
            gamma = tf.ones(tf.shape(loss)) - tf.to_float(loss < max_loss) * tf.ones(tf.shape(loss))
            pi = tf.reduce_sum(tf.reshape(gamma, (x.shape[1], -1, self._k)), reduction_indices=1)
            pi /= tf.reduce_sum(pi, reduction_indices=1, keep_dims=True)

        else:
            raise ValueError('Unknown e_type: "{}"'.format(self._e_step))

        pi = tf.reshape(pi, (1, -1, 1, 1, 1, self._k))

        return gamma, pi

    @staticmethod
    def salt_and_pepper_noise(data, probability, ratio):
        pepper = tf.zeros(tf.shape(data))

        r = tf.random_uniform(tf.shape(data))
        salt = tf.to_float(r >= 1.0 - probability * ratio)

        noisy_data = salt + pepper

        return noisy_data

    @staticmethod
    def gaussian_noise(data, probability, mu, sigma):
        data_copy = tf.identity(data)

        r = tf.random_uniform(tf.shape(data))
        noise_ind = tf.to_float(r >= 1.0 - probability)      # sparse with 1 for noise
        non_noise_ind = tf.to_float(r < 1.0 - probability)   # sparse with 1 for no noise

        noise = noise_ind * tf.random_normal(tf.shape(data), mean=mu, stddev=sigma)
        noisy_data = noise + non_noise_ind * data_copy

        return noisy_data


def get_likelihood(prediction, targets, gamma, distribution):
    if distribution == 'binomial':
        log_loss = targets * np.log(prediction.clip(1e-6, 1 - 1e-6)) + \
                   (1 - targets) * np.log((1 - prediction).clip(1e-6, 1 - 1e-6))
    elif distribution == 'gaussian':
        sigma = 0.1  # todo make dynamic
        loss = (1 / np.sqrt((2 * np.pi * sigma ** 2))) * np.exp(-(targets - prediction) ** 2 / (2 * sigma ** 2))
        log_loss = np.log(loss.clip(1e-6, 1 - 1e-6))
    else:
        raise ValueError('Unknown distribution_type: "{}"'.format(distribution))

    return np.sum(log_loss * gamma, axis=(0,) + tuple(range(2, len(prediction.shape))))


def get_initial_groups(k, dims, init_type, rnd, low=.25, high=.75, hard_assign=False):
    shape = (1, dims[0], dims[1], dims[2], 1, k)  # (T, B, H, W, C, K)
    if init_type == 'spatial':
        assert k == 3
        instace_group_channels = np.zeros((dims[1], dims[2], 3))
        instace_group_channels[:, :, 0] = np.linspace(0, 0.5, dims[1])[:, None]
        instace_group_channels[:, :, 1] = np.linspace(0, 0.5, dims[2])[None, :]
        instace_group_channels[:, :, 2] = 1.0 - instace_group_channels.sum(2)
        instace_group_channels = instace_group_channels.reshape((dims[1], dims[2], 1, 3))

        group_channels = np.zeros(shape)
        group_channels[:, :] = instace_group_channels
    elif init_type == 'gaussian':
        group_channels = np.abs(rnd.randn(*shape))
        group_channels /= group_channels.sum(5)[..., None]
    elif init_type == 'uniform':
        group_channels = rnd.uniform(low, high, size=shape)
        group_channels /= group_channels.sum(5)[..., None]
    else:
        raise ValueError('Unknown init_type "{}"'.format(init_type))

    if hard_assign:
        max_group_channels = np.max(group_channels, axis=5)
        group_channels = np.ones_like(group_channels) - np.array(group_channels < max_group_channels[..., np.newaxis], dtype=float)

    return group_channels


def evaluate_groups(true_groups, predicted):
    scores, confidences = [], []
    for i in range(true_groups.shape[0]):
        true_group, pred = true_groups[i], predicted[i]

        idxs = np.where(true_group != 0.0)
        scores.append(adjusted_mutual_info_score(true_group[idxs], pred.argmax(1)[idxs]))
        confidences.append(np.mean(pred.max(1)[idxs]))

    return scores, confidences