"""
================================LICENSE======================================
Copyright (c) 2015 Chirag Mello & Mario Tambos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
================================LICENSE======================================
"""
import types
import unittest

import numpy as np
import numexpr as ne

from collections import defaultdict
from functools import partial

from spatialpooler import common
from spatialpooler.utils import RingBuffer


class CommonTest(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.columns = np.zeros(shape=(2, 2, 2, 2))
        self.distances =\
            np.asarray([
                        [
                         [[0., 1.], [1., np.sqrt(2)]],
                         [[1., 0.], [np.sqrt(2), 1.]]
                        ],
                        [
                         [[1., np.sqrt(2)], [0., 1.]],
                         [[np.sqrt(2), 1.], [1., 0.]]
                        ]
                       ])

    def test_initialise_synapses(self):
        shape = (2, 2, 2, 2)
        p_connect = 0.5
        connect_threshold = 0.2
        cols, dists = common.initialise_synapses(shape, p_connect,
                                                 connect_threshold)
        self.assertIsInstance(cols, np.ndarray)
        self.assertIsInstance(dists, np.ndarray)
        self.assertEqual(cols.shape, shape)
        expr = ne.evaluate('0 <= cols')
        self.assertTrue((expr | np.isnan(cols)).all())
        expr = ne.evaluate('cols <= 1')
        self.assertTrue((expr | np.isnan(cols)).all())
        self.assertListEqual(dists.tolist(), self.distances.tolist())

    def test_inhibit_columns(self):
        inhibition_area = 2*np.pi
        overlap = np.array([[2, 1], [2, 1]])
        part_deque = partial(RingBuffer, input_array=np.zeros(2), copy=True)
        activity = defaultdict(part_deque)
        desired_activity = 1
        active, activity =\
            common.inhibit_columns(self.columns, self.distances,
                                   inhibition_area, overlap, activity,
                                   desired_activity)
        self.assertIsInstance(active, np.ndarray)
        self.assertEqual(active.shape, (2, 2))
        self.assertIsInstance(activity, defaultdict)
        self.assertListEqual(active.tolist(), [[True, False], [True, False]])
        self.assertIsInstance(activity[0, 0], RingBuffer)
        self.assertListEqual(list(activity[0, 0]), [0, 1])
        self.assertListEqual(list(activity[1, 0]), [0, 1])
        self.assertEqual(len(activity), 4)

    def test_calculate_inhibition_area(self):
        inhibition_area = common.update_inhibition_area(self.columns, 0.2)
        self.assertIsInstance(inhibition_area, types.FloatType)
        self.assertAlmostEqual(inhibition_area, 0.78539816)

    def test_calculate_min_activity(self):
        active = np.array([[1, 1], [1, 1]])
        inhibition_area = np.pi
        part_deque = partial(RingBuffer, input_array=np.zeros(2), copy=True)
        activity = defaultdict(part_deque)
        activity[0, 0].extend([1, 1])
        activity[0, 1].extend([1])
        activity[1, 0].extend([1])
        activity[1, 1].extend([1])
        min_activity_threshold = 1
        min_activity =\
            common.calculate_min_activity(self.columns, active, self.distances,
                                          inhibition_area, activity,
                                          min_activity_threshold)
        self.assertIsInstance(min_activity, np.ndarray)
        self.assertEqual(min_activity.shape, (2, 2))
        self.assertListEqual(min_activity.tolist(), [[1., 2.], [2., 1.]])

    def test_test_for_convergence(self):
        # Test True
        synapses_modified = np.zeros(10, dtype=np.bool)
        converged = common.test_for_convergence(synapses_modified)
        self.assertIsInstance(converged, types.BooleanType)
        self.assertTrue(converged)

        # Test False
        synapses_modified = np.ones(10, dtype=np.bool)
        converged = common.test_for_convergence(synapses_modified)
        self.assertIsInstance(converged, types.BooleanType)
        self.assertFalse(converged)


if __name__ == '__main__':
    unittest.main()
