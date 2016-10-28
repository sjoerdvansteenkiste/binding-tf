#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import os
import numpy as np


def save_image(filename, image_array):
    import scipy.misc
    scipy.misc.imsave(filename, image_array)


def delete_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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
