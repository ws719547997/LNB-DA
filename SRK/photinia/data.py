#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-24
"""

import collections
import queue
import random
import threading

import numpy as np


class DataSource(object):
    """DataSource
    """

    def next_batch(self, size=0):
        """Get a batch of data.

        :param size: Batch size. Default is zero, which means extract all data.
        :return: Tuple of np.array.
        """
        raise NotImplementedError()


class Dataset(DataSource):
    """Dataset
    """

    def __init__(self,data):
        """Construct a dataset.

        :param data: Tuple of list, np.array or any iterable objects.
        :param dtype: Data type.
        """
        self._data = data
        self._size = len(self._data)
        self._num_comp =2
        self._start = 0
        self._loop = 0

    @property
    def size(self):
        return self._size

    @property
    def start(self):
        return self._start

    @property
    def loop(self):
        return self._loop

    def next_batch(self, size=0):
        batch = self._next_batch(size)
        if size == 0:
            return batch
        # real_size = len(batch[0])
        # while real_size < size:
        #     batch1 = self._next_batch(size - real_size)
        #     batch = tuple(np.concatenate((batch, batch1), 0))
        #     real_size = len(batch[0])
        return batch

    def _next_batch(self, size=0):
        if size <= 0:
            return self.all()
        end = self._start + size
        if end < self._size:
            batch = self._data[self._start:end].copy()
            self._start += size
        else:
            batch = self._data[self._start:end].copy()
            self._start = 0
            self._loop += 1
        return batch


    def all(self):
        return self._data[0:self._size].copy()
