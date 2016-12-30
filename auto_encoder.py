#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import tensorflow as tf
from utils import parse_activation_function


class AutoEncoder(object):
    def __init__(self, hidden_sizes, input_size, e_activations, d_activations, tied=False):

        _hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self._sizes = [input_size]
        self._sizes.extend(_hidden_sizes)

        self._tied = tied
        _e_activations = e_activations if isinstance(e_activations, list) else [e_activations]
        self._e_activations = parse_activation_function(_e_activations)
        _d_activations = d_activations if isinstance(d_activations, list) else [d_activations]
        self._d_activations = list(reversed(parse_activation_function(_d_activations)))

    def __call__(self, inputs, reuse=None):

        # encode
        code = inputs
        for i in range(len(self._sizes)-1):
            with tf.variable_scope(str(i), reuse=reuse):
                code = self._encoding_layer(code, self._sizes[i], self._sizes[i+1], self._e_activations[i], reuse=reuse)

        # decode
        outputs = code
        for i in reversed(range(len(self._sizes)-1)):
            with tf.variable_scope(str(i), reuse=reuse):
                outputs = self._decoding_layer(outputs, self._sizes[i+1], self._sizes[i], self._d_activations[i], reuse=reuse)

        return outputs, {'code': code}

    def _encoding_layer(self, layer_inputs, input_size, layer_size, activation, reuse=None):
        with tf.variable_scope('Encode', reuse=reuse):
            W = tf.get_variable('W', shape=[layer_size, input_size])
            bias = tf.get_variable('bias', shape=[layer_size])

        return activation(tf.matmul(layer_inputs, W, transpose_b=True) + bias)

    def _decoding_layer(self, layer_inputs, input_size, layer_size, activation, reuse=None):
        with tf.variable_scope('Decode', reuse=reuse):
            bias = tf.get_variable('bias', shape=[layer_size])

        if self._tied:
            with tf.variable_scope('Encode', reuse=True):
                W = tf.transpose(tf.get_variable('W'))
        else:
            with tf.variable_scope('Decode', reuse=reuse):
                W = tf.get_variable('W', shape=[layer_size, input_size])

        return activation(tf.matmul(layer_inputs, W, transpose_b=True) + bias)
