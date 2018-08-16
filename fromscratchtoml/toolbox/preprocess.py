#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

from fromscratchtoml import np
import numpy

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_onehot(y):

    try:
        import cupy
        if isinstance(y, cupy.core.core.ndarray):
            y = np.asnumpy(y)
    except ImportError:
        pass

    unq, _ = numpy.unique(y, return_inverse=True)

    a = np.zeros((len(y), len(unq)))
    for i in range(len(y)):
        a[i][int(y[i])] = 1.

    return a


def vocab_one_hot(Y, vocab_size):
    temp = np.zeros((Y.shape[0], Y.shape[1], vocab_size))

    for i, _ in enumerate(Y):
        temp[i] = np.eye(vocab_size)[Y[i]]

    return temp
