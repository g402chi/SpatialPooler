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

import os
import os.path as path

from PIL import Image
import numpy as np


class RingBuffer(np.ndarray):
    'A multidimensional ring buffer.'

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def extend(self, xs):
        'Adds array xs to the ring buffer. If xs is longer than the ring '
        'buffer, the last len(ring buffer) of xs are added the ring buffer.'
        xs = np.asarray(xs)
        if self.shape[1:] != xs.shape[1:]:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "xs.shape={}".format(self.shape, xs.shape))
        len_self = len(self)
        len_xs = len(xs)
        if len_self <= len_xs:
            xs = xs[-len_self:]
            len_xs = len(xs)
        else:
            self[:-len_xs] = self[len_xs:]
        self[-len_xs:] = xs

    def append(self, x):
        'Adds element x to the ring buffer.'
        x = np.asarray(x)
        if self.shape[1:] != x.shape:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "xs.shape={}".format(self.shape, x.shape))
        self[:-1] = self[1:]
        self[-1] = x


def iter_columns(columns):
    """
    Go through the HTM column's matrix, yielding at each element the
    element's coordinates together with the synapse's permanence matrix
    associated with that particular column.
    :param columns: the matrix of HTM columns.
    :return: yields a tuple (y, x, syn_matrix), where y is the row index,
             x is the column index and syn_matrix is synapse's permanence
             matrix, all three of a particular HTM column.
    """
    # enumerate(collection) yields a tuple (index, collection_element)
    # for each element in the collection
    for y, col_row in enumerate(columns):
        for x, syn_matrix in enumerate(col_row):
            yield y, x, syn_matrix


def iter_synapses(synapses, only_potential=True):
    """
    Go through the matrix of synapse's permanences associated with a HTM
    column, yielding at each element the element's coordinates together
    with the synapse's permanence value.
    :param synapses: the matrix of synapse's permanences.
    :return: yields a tuple (y, x, syn_perm), where y is the row index,
             x is the column index and syn_perm is synapse's permanence
             value, all three of a particular HTM synapse.
    """
    # enumerate(collection) yields a tuple (index, collection_element)
    # for each element in the collection
    for y, syn_row in enumerate(synapses):
        for x, syn_perm in enumerate(syn_row):
            if not only_potential or not np.isnan(syn_perm):
                yield y, x, syn_perm


def iter_neighbours(columns, y, x, distances, inhibition_radius):
    """
    Go through the matrix of synapse's permanences associated with a HTM
    column, yielding at each element the element's coordinates together
    with the synapse's permanence value.
    :param columns: the matrix of HTM columns.
    :param y: the row index of the HTM column whose neighbours we are looking
              for.
    :param x: the column index of the HTM column whose neighbours we are
              looking for.
    :param distances: an array of shape (m, n, m, n). Each element
                      distance[a, b, c, d] stores the euclidean distance from
                      the HTM column columns[a, b] to the HTM column
                      columns[c, d].
    :param inhibition_radius: the inhibitionArea parameter of the BSP
                              algorithm.
    :return: yields a tuple (y, x, syn_matrix), where y is the row index,
             x is the column index and syn_matrix is synapse's permanence
             matrix, all three of a particular HTM column that is neighbour of
             columns[y, x].
    """
    # Create a boolean array of shape (columns.shape[0], columns.shape[1]).
    # Each element neighbours[a, b] is True if the euclidean distance from
    # (y, x) to (a, b) is inside the inhibition_radius and False otherwise.
    neighbours = distances[y, x] <= inhibition_radius
    # For each column columns[u, v], ...
    for u, v, syn_matrix in iter_columns(columns):
        # if columns[u, v] is a neighbour of columns[y, x], ...
        if neighbours[u, v] and (y, x) != (u, v):
            # yield the coordinates and synapse's matrix of columns[u, v]
            yield u, v, syn_matrix


def extract_patches(images, patch_shape, patches_nr=None,
                    to_bits=False, threshold_method='mean'):
    """
    Extracts patches_nr patches of patch_shape shape from the images. The
    patches are taken randomly from the whole set of images.
    :param images: array of shape (l,m,n) where l is the index of an image,
                   m is the row index in an image and n is the columns index in
                   an image.
    :param patch_shape: tuple (p,q) of two elements specifying the number of
                        rows (p) and columns (q) in a patch.
    :param patches_nr: amount of patches to generate.
    :param to_bits: if True, convert patches' pixels to binary values. The
                    average pixel value per patch will be used as the
                    threshold, everything above it is 1, everything below it
                    is 0.
    :param threshold_method: in the case of conversion to bits, the method
                             used for determining the threshold of selection.
                             Possible values 'mean' or 'median'.
    :return: array of shape (patches_nr, patch_shape[0], patch_shape[1])
    """
    assert threshold_method in ('mean', 'median')
    # get the shape (m, n) of each image
    img_shape = images.shape[1:]
    # calculate what is the maximum index a patch can start at.
    # e.g. if an image is 10x10 pixels, indexed from 0 to 9, and the patches
    # are 3x3 pixels, then x_high = y_high = 7
    x_high = img_shape[1] - patch_shape[1]
    y_high = img_shape[0] - patch_shape[0]
    # if patches_nr is not passed, generate as many patches as possible.
    if patches_nr is None:
        patches_per_img = img_shape[0] - patch_shape[0]
        patches_per_img *= img_shape[1] - patch_shape[1] + 1
        patches_nr = patches_per_img * images.shape[0]
    patches_shape = (patches_nr, patch_shape[0], patch_shape[1])
    # initialize the array to store the patches extracted
    patches = np.zeros(patches_shape)
    if to_bits:
        patches = patches.astype(np.bool)

    i = 0
    while i < patches_nr:
        # pick an image at random
        img_index = np.random.randint(low=0, high=images.shape[0])
        # pick a start row at random
        y_from = np.random.randint(low=0, high=y_high + 1)
        # calculate the end row
        y_to = y_from + patch_shape[0]
        # pick a column at random
        x_from = np.random.randint(low=0, high=x_high + 1)
        # calculate the end column
        x_to = x_from + patch_shape[1]
        # extract patch from the image picked;
        patch = images[img_index, y_from:y_to, x_from:x_to]
        # if is necessary to convert to bits, ...
        if to_bits:
            # calculate the patch's threshold, ...
            if threshold_method == 'mean':
                threshld_lum = patch.mean()
            elif threshold_method == 'median':
                threshld_lum = patch.mean()

            # assign true to everything above the mean, ...
            patch[patch >= threshld_lum] = True
            # assign false to everything below the mean, ...
            patch[patch < threshld_lum] = False
        # if all the pixels in the patch are the same, ...
        if len(np.unique(patch)) > 1:
            # store the patch in the return array.
            patches[i] = patch
            i += 1

    return patches


def load_images(images_path, extensions=('.png',), img_shape=(256, 256)):
    """
    Loads images from images_path into a np.array.
    :param images_path: path to the directory containing the images. The search
                        is not recursive. The images will be converted to
                        grayscale mode w/o alpha channel.
    :param extensions: file extensions to be returned by the search. The
                       extensions must start with a '.' and be lowercase.
    :param images_path: shape of the images. All images found will be reshaped
                        according to this parameter.
    :return: a tuple (np.array of shape
            (len(image files), img_shape[0], img_shape[1]), list of file paths)
    """
    # traverse images_path directory and get all files matching the extensions
    image_files = [path.join(images_path, f) for f in os.listdir(images_path)
                   if (path.isfile(path.join(images_path, f)) and
                       path.splitext(f)[1].lower() in extensions)]
    # create the return array
    images = np.empty((len(image_files), img_shape[0], img_shape[1]))

    # load each file found, ...
    for i, image_file in enumerate(image_files):
        # convert it to grayscale using the formula:
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        # where R, G and B are each pixel's RGB values, ...
        img = Image.open(image_file).convert('L')
        # resize it to the correct dimensions ...
        img = img.resize(img_shape, Image.LANCZOS)
        # and add it to the return array
        images[i] = np.asarray(img)

    return images, image_files


def read_input(images):
    """
    Randomly yields an image without replacement from **images**.
    :param images: np.array of images
    :return: an iterator. Yields a tuple (np.array with an image, index of the
             image in **images**)
    """
    # create an index array
    images_order = np.arange(len(images))
    # shuffle the index
    np.random.shuffle(images_order)
    # iterate over the shuffled index
    for i, image_index in enumerate(images_order):
        yield i, images[image_index], image_index


def euclidean_dist(matrx, vectr):
    """
    Calculates the euclidean distance from each row of matrx to vectr.
    :param matrx: an array of shape (m, n)
    :param vectr: an array of shape (n,)
    :return: an array of shape (m,)
    """
    # coerce matrx to be of type np.array
    matrx = np.asarray(matrx)
    # make sure matrx has the correct shape
    assert len(matrx.shape) == 2
    # coerce vectr to be of type np.array
    vectr = np.asarray(vectr)
    # make sure vectr has the correct shape
    assert len(vectr.shape) == 1
    # make sure the column size of matrx is the same as the size of vectr
    assert matrx.shape[1] == vectr.shape[0]
    # calculate the element-wise squared difference between matrx and vectr
    # this results in an array of shape (m, n)
    dist = (matrx - vectr)**2
    # sum along columns: c1 + c2 + ... + cn
    # this results in an array of shape (m,)
    dist = np.sum(dist, axis=1)
    # return the element-wise square root of dist
    return np.sqrt(dist)
