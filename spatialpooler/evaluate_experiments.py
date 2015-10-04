#!/usr/bin/env python
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
from collections import defaultdict
import cPickle as pickle
from functools import partial
from pprint import pprint
import os

import numpy as np
import matplotlib.pyplot as plt

from ASP import calculate_overlap as asp_overlap
from BSP import calculate_overlap as bsp_overlap
from common import calculate_distances, inhibit_columns, update_inhibition_area
from utils import (extract_patches, load_images, RingBuffer, read_input,
                   rebuild_imgs_from_patches, grayscale_to_bits)


def calculate_lifetime_kurtosis(activations):
    # Get the act_shape of the activations array
    act_shape = activations.shape
    # Reshape the activations array from 3D to 2D by flattening the columns
    # matrix.
    activations = activations.reshape(act_shape[0],
                                      act_shape[1] * act_shape[2])
    # For each column, calculate its activation'S mean  for all images.
    col_mean_act = np.nanmean(activations, axis=0)
    # For each column, calculate its activation's standard deviation for all
    # images.
    col_std_act = np.nanstd(activations, axis=0)
    # For each column, normalize its activation to mean 0 and std 1.
    col_life_kurtosis = (activations - col_mean_act) / col_std_act
    # For each column, calculate its kurtosis.
    col_life_kurtosis = np.power(col_life_kurtosis, 4)
    col_life_kurtosis = np.nansum(col_life_kurtosis, axis=0)/act_shape[0] - 3

    # Return the mean kurtosis over all columns
    return np.nansum(col_life_kurtosis)/col_life_kurtosis.shape[0]


def calculate_population_kurtosis(activations):
    # Get the act_shape of the activations array
    act_shape = activations.shape
    # Reshape the activations array from 3D to 2D by flattening the columns
    # matrix.
    activations = activations.reshape(act_shape[0],
                                      act_shape[1] * act_shape[2])
    # For each image, calculate its activation'S mean  for all columns.
    img_mean_act = np.nanmean(activations, axis=1)
    # For each image, calculate its activation's standard deviation for all
    # images.
    img_std_act = np.nanstd(activations, axis=1)
    # For each image, normalize its activation to mean 0 and std 1.
    img_pop_kurtosis = ((activations.T - img_mean_act) / img_std_act).T
    # For each image, calculate its kurtosis.
    img_pop_kurtosis = np.power(img_pop_kurtosis, 4)
    img_pop_kurtosis = (np.nansum(img_pop_kurtosis, axis=1) /
                        activations.shape[1] - 3)

    # Return the mean kurtosis over all images
    return np.nansum(img_pop_kurtosis)/img_pop_kurtosis.shape[0]


def calculate_code_stats(activations):
    # Get the act_shape of the activations array
    act_shape = activations.shape
    # Reshape the activations array from 3D to 2D by flattening the columns
    # matrix.
    activations = activations.reshape(act_shape[0],
                                      act_shape[1] * act_shape[2])
    activations = activations > 0
    b = (np.ascontiguousarray(activations)
         .view(np.dtype(
                        (np.void,
                         activations.dtype.itemsize *
                         activations.shape[1]))))
    _, idx = np.unique(b, return_index=True)
    unique_act = activations[idx]
    duplicates_nr = unique_act.shape[0]
    dups_perc = duplicates_nr/act_shape[0]
    zero_nr = np.argwhere(~(activations > 0).any(axis=1)).size
    zero_perc = zero_nr/act_shape[0]
    mean_code_len = activations.sum()/act_shape[0]

    return dups_perc, zero_perc, mean_code_len


def reconstruct_images(alg, images, columns, connect_threshold,
                       desired_activity_mult, min_overlap, img_shape,
                       out_dir=None):
    cols_shape = columns.shape
    # Initialize boost matrix.
    boost = np.ones(shape=cols_shape[:2])
    part_rbuffer = partial(RingBuffer,
                           input_array=np.zeros(1, dtype=np.bool), copy=True)
    # Initialize activity dictionary.
    activity = defaultdict(part_rbuffer)
    if alg == 'bsp':
        # Initialize overlap_sum dictionary.
        overlap_sum = defaultdict(part_rbuffer)

    distances = calculate_distances(cols_shape)
    # Calculate the inhibition_area parameter.
    inhibition_area = update_inhibition_area(columns, connect_threshold)
    # Calculate the desired activity in a inhibition zone.
    desired_activity = desired_activity_mult * inhibition_area

    reconstructions = np.zeros_like(images)
    # Initialize the activations matrix. This will be used to calculate the
    # population and lifetime kurtoses.
    activations = np.zeros(shape=(images.shape[0], cols_shape[0],
                                  cols_shape[1]), dtype=np.int)

    for i, image, _ in read_input(images):
        if alg == 'bsp':
            overlap, _ = bsp_overlap(image, columns, min_overlap,
                                     connect_threshold, boost, overlap_sum)
        elif alg == 'asp':
            # calculate the overlap of the columns with the image.
            # (this is a simple count of the number of its connected synapses
            # that are receiving active input (p. 3)), ...
            overlap = asp_overlap(image, columns, min_overlap,
                                  connect_threshold, boost)
        # force sparsity by inhibiting columns, ...
        active, _ = inhibit_columns(columns, distances, inhibition_area,
                                    overlap, activity, desired_activity)
        # set reconstructions[i][y, x] to (sic, p. 7):
        #    "[...] the linear superposition of the
        #     connected synapses of the columns that become active
        #     when the input is present [...]"
        # first, generate a copy of the columns array, where all the synapses
        # of all inactive columns are set to 0, ...
        active_cols = np.where(active, columns,
                               np.zeros(shape=(cols_shape[2], cols_shape[3])))
        # and then set reconstructions[i] to the linear superposition of the
        # synapses of the active columns.
        reconstructions[i] = np.nansum(active_cols, axis=(0, 1))
        # Store the post-inhibition overlap activity of each column as the
        # sum of the overlap of the active columns.
        activations[i] = np.nansum(columns, axis=(2, 3))

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        # Rebuild the 256x 256 images from the 16x16 patches.
        reconstructions = rebuild_imgs_from_patches(reconstructions, img_shape)
        # Scale the pixel values to the range [0, 1].
        reconstructions = reconstructions - reconstructions.min()
        reconstructions /= reconstructions.max() - reconstructions.min()
        # Scale the pixel values to the range [0, 255].
        reconstructions *= 255
        for i, reconstruct_img in enumerate(reconstructions):
            with open(os.path.join(out_dir, 'rec_img_%d.png' % i), 'wb') as fp:
                plt.imsave(fname=fp, arr=reconstruct_img, cmap='gray', )
    return activations


def save_activations(activations, columns_file):
    dir_name = os.path.dirname(columns_file)
    base_name = os.path.basename(columns_file)
    act_out_file = os.path.join(dir_name, 'activations_%s' % base_name)
    with open(act_out_file, 'wb') as fp:
        pickle.dump(activations, fp)


def calculate_print_stats(activations, alg):
    pop_kurtosis = calculate_population_kurtosis(activations)
    lif_kurtosis = calculate_lifetime_kurtosis(activations)
    dups, zero, code_len = calculate_code_stats(activations)
    pprint("%s lifetime kurtosis: %0.4f" % (alg, lif_kurtosis))
    pprint("%s population kurtosis: %0.4f" % (alg, pop_kurtosis))
    pprint("%s %% code duplicates: %0.4f%%" % (alg, dups*100))
    pprint("%s %% code zero length: %0.4f%%" % (alg, zero*100))
    pprint("%s mean code length: %0.4f" % (alg, code_len))


if __name__ == '__main__':
    import sys

    # Check whether the --imgs_dir command line parameter was
    # provided.
    imgs_dir = None
    if '--imgs_dir' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--imgs_dir')
        imgs_dir = sys.argv[arg_index + 1]
    else:
        sys.exit('The --imgs_dir parameter is mandatory.')

    # Check whether the --bsp_columns_file command line parameter was
    # provided.
    bsp_columns_file = None
    if '--bsp_columns_file' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--bsp_columns_file')
        bsp_columns_file = sys.argv[arg_index + 1]

    # Check whether the --asp_columns_file command line parameter was
    # provided.
    asp_columns_file = None
    if '--asp_columns_file' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--asp_columns_file')
        asp_columns_file = sys.argv[arg_index + 1]

    # Check whether the --bsp_out_dir command line parameter was
    # provided.
    bsp_out_dir = None
    if '--bsp_out_dir' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--bsp_out_dir')
        bsp_out_dir = sys.argv[arg_index + 1]
    # Check whether the --asp_out_dir command line parameter was
    # provided.
    asp_out_dir = None
    if '--asp_out_dir' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--asp_out_dir')
        asp_out_dir = sys.argv[arg_index + 1]

    if bsp_columns_file is None and asp_columns_file is None:
        sys.exit('One or both of the --bsp_columns_file'
                 ' --asp_columns_file must be provided.')

    images, _ = load_images(imgs_dir, extensions=('.jpg',),
                            img_shape=(256, 256))
    patches = extract_patches(images, (16, 16), images.shape[0]*256,
                              randomize=False)

    if asp_columns_file is not None:
        with open(asp_columns_file, 'rb') as fp:
            asp_columns = pickle.load(fp)
        asp_activations =\
            reconstruct_images(alg='asp', images=patches, columns=asp_columns,
                               connect_threshold=0.2,
                               desired_activity_mult=0.05, min_overlap=3,
                               img_shape=(256, 256), out_dir=asp_out_dir)
        calculate_print_stats(asp_activations, alg='ASP')
        save_activations(asp_activations, asp_columns_file)

    if bsp_columns_file is not None:
        with open(bsp_columns_file, 'rb') as fp:
            bsp_columns = pickle.load(fp)
        binary_patches = grayscale_to_bits(patches)
        bsp_activations =\
            reconstruct_images(alg='bsp', images=binary_patches,
                               columns=bsp_columns,
                               connect_threshold=0.2,
                               desired_activity_mult=0.05, min_overlap=3,
                               img_shape=(256, 256), out_dir=bsp_out_dir)
        calculate_print_stats(bsp_activations, alg='BSP')
        save_activations(bsp_activations, bsp_columns_file)
