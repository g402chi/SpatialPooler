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

from spatialpooler import BSP
from collections import defaultdict, deque
from functools import partial


class BSPTest(unittest.TestCase):
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
        cols, dists = BSP.initialise_synapses(shape, p_connect,
                                              connect_threshold)
        self.assertIsInstance(cols, np.ndarray)
        self.assertIsInstance(dists, np.ndarray)
        self.assertEqual(cols.shape, shape)
        expr = ne.evaluate('0 <= cols')
        self.assertTrue((expr | np.isnan(cols)).all())
        expr = ne.evaluate('cols <= 1')
        self.assertTrue((expr | np.isnan(cols)).all())
        self.assertListEqual(dists.tolist(), self.distances.tolist())

    def test_calculate_overlap(self):
        input_vector = np.array([[1, 0], [0, 0]])
        self.columns[0, 0, 0, 0] = 1
        min_overlap = 1
        connect_threshold = 0.1
        boost = np.array([[1, 1], [1, 1]])
        part_deque = partial(deque, maxlen=2)
        overlap_sum = defaultdict(part_deque)
        overlap, overlap_sum =\
            BSP.calculate_overlap(input_vector, self.columns, min_overlap,
                                  connect_threshold, boost, overlap_sum)
        self.assertIsInstance(overlap, np.ndarray)
        self.assertEqual(overlap.shape, (2, 2))
        self.assertIsInstance(overlap_sum, defaultdict)
        self.assertListEqual(overlap.tolist(), [[1., 0.], [0., 0.]])
        self.assertIsInstance(overlap_sum[0, 0], deque)
        self.assertListEqual(list(overlap_sum[0, 0]), [1])

    def test_inhibit_columns(self):
        inhibition_area = 2
        overlap = np.array([[2, 1], [2, 1]])
        part_deque = partial(deque, maxlen=2)
        activity = defaultdict(part_deque)
        desired_activity = 1
        active, activity =\
            BSP.inhibit_columns(self.columns, self.distances,
                                inhibition_area, overlap, activity,
                                desired_activity)
        print active, activity
        self.assertIsInstance(active, np.ndarray)
        self.assertEqual(active.shape, (2, 2))
        self.assertIsInstance(activity, defaultdict)
        self.assertListEqual(active.tolist(), [[True, False], [True, False]])
        self.assertIsInstance(activity[0, 0], deque)
        self.assertListEqual(list(activity[0, 0]), [1])
        self.assertListEqual(list(activity[1, 0]), [1])
        self.assertEqual(len(activity), 2)

    def test_calculate_inhibition_area(self):
        inhibition_area = BSP.update_inhibition_area(self.columns)
        self.assertIsInstance(inhibition_area, np.ndarray)
        self.assertEqual(inhibition_area.shape, (2, ))
        self.assertAlmostEqual(inhibition_area[0], 0.78539816)
        self.assertAlmostEqual(inhibition_area[1], 0.78539816)

    def test_calculate_min_activity(self):
        active = np.array([[1, 1], [1, 1]])
        inhibition_area = 1
        part_deque = partial(deque, maxlen=2)
        activity = defaultdict(part_deque)
        activity[0, 0].extend([1, 1])
        activity[0, 1].extend([1])
        activity[1, 0].extend([1])
        activity[1, 1].extend([1])
        min_activity_threshold = 1
        min_activity =\
            BSP.calculate_min_activity(self.columns, active, self.distances,
                                       inhibition_area, activity,
                                       min_activity_threshold)
        self.assertIsInstance(min_activity, np.ndarray)
        self.assertEqual(min_activity.shape, (2, 2))
        self.assertListEqual(min_activity.tolist(), [[1., 2.], [2., 1.]])

    def test_learn_synapse_connections(self):
        active = np.array([[1, 1], [1, 1]])
        input_vector = np.array([[1, 0], [0, 0]])
        p_inc = 0.1
        p_dec = 0.1
        part_deque = partial(deque, maxlen=2)
        activity = defaultdict(part_deque)
        overlap_sum = defaultdict(part_deque)
        min_activity = np.array([[1, 0], [0, 0]])
        boost = np.array([[1, 1], [1, 1]])
        b_inc = 1
        p_mult = 1
        columns, synapse_modified =\
            BSP.learn_synapse_connections(self.columns, active,
                                          input_vector, p_inc, p_dec,
                                          activity, overlap_sum,
                                          min_activity, boost, b_inc,
                                          p_mult)
        print columns, synapse_modified
        self.assertIsInstance(columns, np.ndarray)
        self.assertEqual(columns.shape, (2, 2, 2, 2))
        self.assertListEqual(columns.tolist(), [
                                                [
                                                 [[0.1, 0.], [0., 0.]],
                                                 [[0.1, 0.], [0., 0.]]
                                                ],
                                                [
                                                 [[0.1, 0.], [0., 0.]],
                                                 [[0.1, 0.], [0., 0.]]
                                                ]
                                               ])
        self.assertIsInstance(synapse_modified, types.BooleanType)
        self.assertFalse(synapse_modified)

    def test_test_for_convergence(self):
        # Test True
        synapses_modified = np.zeros(10, dtype=np.bool)
        converged = BSP.test_for_convergence(synapses_modified)
        self.assertIsInstance(converged, types.BooleanType)
        self.assertTrue(converged)

        # Test False
        synapses_modified = np.ones(10, dtype=np.bool)
        converged = BSP.test_for_convergence(synapses_modified)
        self.assertIsInstance(converged, types.BooleanType)
        self.assertFalse(converged)

    def test_spatial_pooler(self,):
        images = np.zeros(shape=(10, 2, 2))
        shape = (2, 2, 2, 2)
        p_connect = 0.15
        connect_threshold = 0.2
        p_inc = 0.02
        p_dec = 0.02
        b_inc = 0.005
        p_mult = 0.01
        min_activity_threshold = 0.01
        min_overlap = 3,
        desired_activity_mult = 0.05
        max_iterations = 1000
        cols = BSP.spatial_pooler(images, shape, p_connect,
                                  connect_threshold, p_inc,
                                  p_dec, b_inc, p_mult,
                                  min_activity_threshold, min_overlap,
                                  desired_activity_mult,
                                  max_iterations)
        self.assertIsInstance(cols, np.ndarray)
        self.assertEqual(cols.shape, shape)
        expr = ne.evaluate('0 <= cols')
        self.assertTrue((expr | np.isnan(cols)).all())
        expr = ne.evaluate('cols <= 1')
        self.assertTrue((expr | np.isnan(cols)).all())


if __name__ == '__main__':
    unittest.main()
