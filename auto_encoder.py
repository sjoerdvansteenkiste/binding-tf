#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import tensorflow as tf


class AutoEncoder(object):
    def __init__(self, size, input_size, hidden_activation=tf.sigmoid, output_activation=tf.sigmoid, tied=False):
        self._size = size
        self._input_size = input_size
        self._tied = tied
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation

    def __call__(self, inputs, scope=''):
        with tf.variable_scope(scope + 'Encode'):
            W_enc = tf.get_variable('W', shape=[self._size, self._input_size], dtype=tf.float32)
            bias_enc = tf.get_variable('bias', shape=[self._size], dtype=tf.float32)

            code = self._hidden_activation(tf.matmul(inputs, W_enc, transpose_b=True) + bias_enc)

        with tf.variable_scope(scope + 'Decode'):
            bias_dec = tf.get_variable('bias', shape=[self._input_size], dtype=tf.float32)

            if self._tied:
                outputs = self._output_activation(tf.matmul(code, W_enc) + bias_dec)
            else:
                W_dec = tf.get_variable('W', shape=[self._input_size, self._size], dtype=tf.float32)
                outputs = self._output_activation(tf.matmul(code, W_dec, transpose_b=True) + bias_dec)

        return outputs, {'code': code}
