#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import tensorflow as tf


class AutoEncoder(object):
    def __init__(self, hidden_sizes, input_size, hidden_activation=tf.sigmoid, output_activation=tf.sigmoid,
                 tied=False):

        _hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self._sizes = [input_size]
        self._sizes.extend(_hidden_sizes)

        self._tied = tied
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation

    def __call__(self, inputs):

        # encode
        code = inputs
        for i in range(len(self._sizes)-1):
            with tf.variable_scope(str(i)):
                code = self._encoding_layer(code, self._sizes[i], self._sizes[i+1])

        # decode
        outputs = code
        for i in reversed(range(len(self._sizes)-1)):
            with tf.variable_scope(str(i)):
                outputs = self._decoding_layer(outputs, self._sizes[i+1], self._sizes[i], self._tied)

        return outputs, {'code': code}

    def _encoding_layer(self, inputs, input_size, size):
        with tf.variable_scope('Encode'):
            W = tf.get_variable('W', shape=[size, input_size], dtype=tf.float32)
            bias = tf.get_variable('bias', shape=[size], dtype=tf.float32)

        return self._hidden_activation(tf.matmul(inputs, W, transpose_b=True) + bias)

    def _decoding_layer(self, inputs, input_size, size, tied):
        with tf.variable_scope('Decode'):
            bias = tf.get_variable('bias', shape=[size], dtype=tf.float32)

        if tied:
            with tf.variable_scope('Encode', reuse=True):
                W = tf.transpose(tf.get_variable('W'))
        else:
            with tf.variable_scope('Decode'):
                W = tf.get_variable('W', shape=[size, input_size], dtype=tf.float32)

        return self._output_activation(tf.matmul(inputs, W, transpose_b=True) + bias)
