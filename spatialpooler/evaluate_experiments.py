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
from functools import partial
from pprint import pprint

import numpy as np

from ASP import calculate_overlap as asp_overlap
from BSP import calculate_overlap as bsp_overlap
from common import calculate_distances, inhibit_columns, update_inhibition_area
from utils import RingBuffer, read_input


def calculate_lifetime_kurtosis(activations):
    # Get the shape of the activations array
    shape = activations.shape
    # Reshape the activations array from 3D to 2D by flattening the columns
    # matrix.
    activations = activations.reshape(shape[0], shape[1] * shape[2])
    # For each column, calculate its activation'S mean  for all images.
    col_mean_act = activations.mean(axis=0)
    # For each column, calculate its activation'S standard devition for all
    # images.
    col_std_act = activations.std(axis=0)
    # For each column, normalize its activation to mean 0 and std 1.
    col_life_kurtosis = (activations - col_mean_act) / col_std_act
    # For each column, calculate its kurtosis.
    col_life_kurtosis = np.power(col_life_kurtosis, 4)
    col_life_kurtosis = col_life_kurtosis.sum(axis=0)/shape[0] - 3

    # Return the mean kurtosis over all columns
    return col_life_kurtosis.sum()/col_life_kurtosis.shape[0]


def calculate_population_kurtosis(activations):
    # Get the shape of the activations array
    shape = activations.shape
    # Reshape the activations array from 3D to 2D by flattening the columns
    # matrix.
    activations = activations.reshape(shape[0], shape[1] * shape[2])
    # For each image, calculate its activation'S mean  for all columns.
    img_mean_act = activations.mean(axis=1)
    # For each image, calculate its activation'S standard devition for all
    # images.
    img_std_act = activations.std(axis=1)
    # For each image, normalize its activation to mean 0 and std 1.
    img_pop_kurtosis = ((activations.T - img_mean_act) / img_std_act).T
    # For each image, calculate its kurtosis.
    img_pop_kurtosis = np.power(img_pop_kurtosis, 4)
    img_pop_kurtosis = img_pop_kurtosis.sum(axis=1)/activations.shape[1] - 3

    # Return the mean kurtosis over all images
    return img_pop_kurtosis.sum()/img_pop_kurtosis.shape[0]


def asp_reconstruct_images(images, columns, connect_threshold,
                           desired_activity_mult, min_overlap):
    shape = columns.shape
    # Initialize boost matrix.
    boost = np.ones(shape=shape[:2])
    part_rbuffer = partial(RingBuffer,
                           input_array=np.zeros(1, dtype=np.bool))
    # Initialize activity dictionary.
    activity = defaultdict(part_rbuffer)

    distances = calculate_distances(shape)
    # Calculate the inhibition_area parameter.
    inhibition_area = update_inhibition_area(columns, connect_threshold)
    # Calculate the desired activity in a inhibition zone.
    desired_activity = desired_activity_mult * inhibition_area

    reconstructions = np.zeros_like(images)

    for i, image, _ in read_input(images):
        # calculate the overlap of the columns with the image.
        # (this is a simple count of the number of its connected synapses
        # that are receiving active input (p. 3)), ...
        overlap = asp_overlap(image, columns, min_overlap,
                              connect_threshold, boost)
        # force sparsity by inhibiting columns, ...
        active, _ = inhibit_columns(columns, distances, inhibition_area,
                                    overlap, activity, desired_activity)
        reconstructions[i] = columns[active].sum()


def bsp_reconstruct_images(images, columns, connect_threshold,
                           desired_activity_mult, min_overlap):
    shape = columns.shape
    # Initialize boost matrix.
    boost = np.ones(shape=shape[:2])
    part_rbuffer = partial(RingBuffer,
                           input_array=np.zeros(1, dtype=np.bool))
    # Initialize activity dictionary.
    activity = defaultdict(part_rbuffer)
    # Initialize overlap_sum dictionary.
    overlap_sum = defaultdict(part_rbuffer)

    distances = calculate_distances(shape)
    # Calculate the inhibition_area parameter.
    inhibition_area = update_inhibition_area(columns, connect_threshold)
    # Calculate the desired activity in a inhibition zone.
    desired_activity = desired_activity_mult * inhibition_area

    reconstructions = np.zeros_like(images)
    for i, image, _ in read_input(images):
        # calculate the overlap of the columns with the image.
        # (this is a simple count of the number of its connected synapses
        # that are receiving active input (p. 3)), ...
        overlap = bsp_overlap(image, columns, min_overlap,
                              connect_threshold, boost, overlap_sum)
        # force sparsity by inhibiting columns, ...
        active, activity = inhibit_columns(columns, distances, inhibition_area,
                                           overlap, activity, desired_activity)
        reconstructions[i] = columns[active].sum()


if __name__ == '__main__':
    import cPickle as pickle
    import sys

    # Check whether the --bsp_activations_file command line parameter was
    # provided.
    bsp_activations_file = None
    if '--bsp_activations_file' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--bsp_activations_file')
        bsp_activations_file = sys.argv[arg_index + 1]

    # Check whether the --asp_activations_file command line parameter was
    # provided.
    asp_activations_file = None
    if '--asp_activations_file' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--asp_activations_file')
        asp_activations_file = sys.argv[arg_index + 1]

    if bsp_activations_file is None and asp_activations_file is None:
        sys.exit('One or both of the --bsp_activations_file'
                 ' --asp_activations_fileand must be provided.')

    if bsp_activations_file is not None:
        with open(bsp_activations_file, 'rb') as fp:
            bsp_activations = pickle.load(fp)
        bsp_pop_kurtosis = calculate_population_kurtosis(bsp_activations)
        bsp_lif_kurtosis = calculate_lifetime_kurtosis(bsp_activations)
        pprint("BSP lifetime kurtosis: %0.4f" % bsp_lif_kurtosis)
        pprint("BSP population kurtosis: %0.4f" % bsp_pop_kurtosis)

    if asp_activations_file is not None:
        with open(asp_activations_file, 'rb') as fp:
            asp_activations = pickle.load(fp)
        asp_pop_kurtosis = calculate_population_kurtosis(asp_activations)
        asp_lif_kurtosis = calculate_lifetime_kurtosis(asp_activations)
        pprint("ASP lifetime kurtosis: %0.4f" % asp_lif_kurtosis)
        pprint("ASP population kurtosis: %0.4f" % asp_pop_kurtosis)
