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

from spatialpooler import utils
import os
import spatialpooler


class UtilsTest(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.columns = np.zeros(shape=(2, 2, 2, 2))
        self.distances =\
            np.asarray([
                        [
                         [[0., 1.], [1., 1.41421356]],
                         [[1., 0.], [1.41421356, 1.]]
                        ],
                        [
                         [[1., 1.41421356], [0., 1.]],
                         [[1.41421356, 1.], [1., 0.]]
                        ]
                       ])
        bsp_path = os.path.dirname(spatialpooler.__file__)
        root_dir = os.path.abspath(os.path.join(bsp_path, os.pardir))
        self.images_path = os.path.join(root_dir, 'images')

    def test_iter_columns(self):
        inst = utils.iter_columns(self.columns)
        self.assertIsInstance(inst, types.GeneratorType)
        for y, x, syn_matrix in inst:
            self.assertIsInstance(y, types.IntType)
            self.assertIn(y, (0, 1))
            self.assertIsInstance(x, types.IntType)
            self.assertIn(x, (0, 1))
            self.assertIsInstance(syn_matrix, np.ndarray)
            self.assertEqual(syn_matrix.shape, (2, 2))

    def test_iter_synapses(self):
        _, _, syn_matrix = utils.iter_columns(self.columns).next()
        inst = utils.iter_synapses(syn_matrix)
        self.assertIsInstance(inst, types.GeneratorType)
        for y, x, perm in inst:
            self.assertIsInstance(y, types.IntType)
            self.assertIn(y, (0, 1))
            self.assertIsInstance(x, types.IntType)
            self.assertIn(x, (0, 1))
            self.assertIsInstance(perm, types.FloatType)

    def test_iter_neighbours(self, ):
        inhibition_area = 1.1
        inst = utils.iter_neighbours(self.columns, 0, 0,
                                     self.distances, inhibition_area)
        self.assertIsInstance(inst, types.GeneratorType)
        for y, x, syn_matrix in inst:
            self.assertIsInstance(y, types.IntType)
            self.assertIsInstance(x, types.IntType)
            self.assertIn((y, x), ((0, 0), (0, 1), (1, 0)))
            self.assertIsInstance(syn_matrix, np.ndarray)
            self.assertEqual(syn_matrix.shape, (2, 2))

    def test_extract_patches(self):
        images = np.array([[range(10)]*10]*100)
        patches = utils.extract_patches(images, (3, 3), patches_nr=15)
        for patch in patches:
            patch_in_image = False
            for image in images:
                for y, row in enumerate(image[:-2]):
                    for x, _ in enumerate(row[:-2]):
                        if (patch == image[y:y+3, x:x+3]).all():
                            patch_in_image = True
                            break
                    if patch_in_image:
                        # patch found, stop iterating over rows
                        break
                if patch_in_image:
                    # patch found, stop iterating over images
                    break
            self.assertTrue(patch_in_image)

    def test_extract_patches_bits(self):
        images = np.array([[range(10)]*10]*100)
        patches = utils.extract_patches(images, (3, 3), patches_nr=15,
                                        to_bits=True)
        for patch in patches:
            self.assertIsInstance(patch, np.ndarray)
            self.assertEqual(patch.dtype, np.bool)

    def test_load_images(self):
        images, image_files = utils.load_images(self.images_path,
                                                extensions=('.jpg',),
                                                img_shape=(256, 256))
        self.assertIsInstance(images, np.ndarray)
        self.assertEqual(images.shape, (64, 256, 256))
        self.assertNotEqual(np.count_nonzero(images), 0)
        self.assertEqual(len(image_files), 64)

    def test_read_input(self):
        images, _ = utils.load_images(self.images_path,
                                      extensions=('.jpg',),
                                      img_shape=(256, 256))
        inst = utils.read_input(images)
        self.assertIsInstance(inst, types.GeneratorType)
        for i, image, image_index in inst:
            self.assertIsInstance(i, types.IntType)
            self.assertIn(i, range(len(images)))
            self.assertIsInstance(image_index, types.IntType)
            self.assertIn(image_index, range(len(images)))
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (256, 256))
            self.assertListEqual(image.tolist(), images[image_index].tolist())

    def test_euclidean_dist(self):
        matrx = np.array([[i, j] for i in range(3) for j in range(2)])
        vectr = [0, 0]
        dist = utils.euclidean_dist(matrx, vectr)
        self.assertIsInstance(dist, np.ndarray)
        self.assertEqual(dist.shape, (6, ))
        self.assertListEqual(dist.tolist(),
                             np.array([0., 1., 1., np.sqrt(2),
                                       2., np.sqrt(5)]).tolist())


if __name__ == '__main__':
    unittest.main()
