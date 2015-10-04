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


Implementation of the augmented spatial pooler according to:

Thornton, John, and Andrew Srbic. "Spatial pooling for greyscale images."
International Journal of Machine Learning and Cybernetics 4, no. 3 (2013):
207-216.

A region is represented by a 4-dimensional numpy array **columns**, such that:

columns[i, j, k, l] == v in [0, 1) => the column mapped to thecoordinates
                                     [i, j] has a *potential* synapse to
                                     coordinates [k, l] with permanence v

Another way of thinking about it is that **columns** is a 2-dimensional array
whose elements are columns, and each column is in turn a 2-dimensional array
of synapses.
"""
from __future__ import division, print_function

from collections import defaultdict
import cPickle as pickle
from datetime import datetime
from functools import partial
from pprint import pprint

import numpy as np
import numexpr as ne

from utils import read_input, iter_columns, iter_synapses
from common import (update_inhibition_area, calculate_min_activity,
                    inhibit_columns, initialise_synapses, test_for_convergence)
from utils import RingBuffer


def calculate_overlap(input_vector, columns, min_overlap, connect_threshold,
                      boost):
    """
    Implements the calculateOverlap function from the paper (p. 3).
    :param input_vector: a single input_vector from the images set.
    :param columns: the 4-dimensional array of HTM columns: a 4-dimensional
                    array of shape *shape* that represents the HTM columns and
                    their synapses; each element columns[a, b, c, d] contains
                    the permanence value of the synapse connecting the column
                    with coordinates (a, b) to input with coordinates (c, d).
    :param min_overlap: the BSP's minOverlap parameter (p. 3). Type: float.
    :param connect_threshold: the BSP's connectThreshold parameter (p. 3).
                              Type: float.
    :param boost: the BSP's boost matrix (pp. 3, 4).
                  It is a matrix of shape (columns.shape[0], columns.shape[1])
    :param overlap_sum: the BSP's overlapSum matrix (p. 4). This parameter is
                        modified and returned. It is a dictionary with tuples
                        (y, x) of HTM column coordinates as keys and deque
                        (a queue implementation) instances as values. The queue
                        for key [a, b] stores a 1 each time the overlap of the
                        column [a, b] was above the minOverlap threshold during
                        the last 1000 iterations.
    :return: a tuple (overlap, overlap_sum). *overlap* is an array of shape
             (columns.shape[0], columns.shape[0]); each element overlap[a, b]
             contains the overlap of the column [a, b] with the input_vector.
             *overlap_sum* is the parameter of the same name; the queue in
             overlap_sum[a, b] will have a 1 pushed into it if the overlap for
             the column [a, b] was above the min_overlap threshold this
             iteration.
    """
    # Initialize the overlap array.
    overlap = np.zeros(columns.shape[:2])
    # for each column ...
    for y, x, syn_matrix in iter_columns(columns):  # @UnusedVariable
        c = (y, x)
        # calculate the overlap as the sum of pixel's values in the
        # input_vector assigned to *connected* synapses and, ...
        # (numexpr.evaluate optimizes and carries out the operations defined in
        # its string argument, it is used here because numpy has some
        # problems when comparing NaNs with numbers)
        active_synapses = ne.evaluate('syn_matrix >= connect_threshold')
        overlap[c] = input_vector[active_synapses].sum()
        # if the overlap is not enough, ...
        if overlap[c] < min_overlap:
            # reset it, but, ...
            overlap[c] = 0
        # if the overlap is enough, ...
        else:
            # then boost it.
            overlap[c] *= boost[c]

    return overlap


def learn_synapse_connections(columns, active, input_vector, p_inc,
                              p_dec, activity, min_activity, boost, b_inc,
                              p_mult, connect_threshold, distances, b_max):
    """
    Calculates the minActivity matrix from the paper (p. 6).
    :param columns: the 4-dimensional array of HTM columns: a 4-dimensional
                    array of shape *shape* that represents the HTM columns and
                    their synapses; each element columns[a, b, c, d] contains
                    the permanence value of the synapse connecting the column
                    with coordinates (a, b) to input with coordinates (c, d).
    :param active: array of shape (columns.shape[0], columns.shape[0]);
                   each element active[a, b] is True if the column [a, b]
                   is active this iteration and False otherwise.
    :param input_vector: the input to be learned. A 2-dimensional array of
                         shape (columns.shape[0], columns.shape[1]).
    :param p_inc: the BSP'perm pInc parameter (p. 4). A float that indicates
                  the amount by which a synapse'perm permanence must be
                  incremented.
    :param p_dec: the BSP'perm pDec parameter (p. 4). A float that indicates
                  the amount by which a synapse'perm permanence must be
                  decremented.
    :param activity: the BSP'perm activity matrix (p. 4).
                     This parameter is modified and returned.
                     It is a dictionary with tuples (y, x) of HTM column
                     coordinates as keys and deque (a queue implementation)
                     instances as values. The queue for key [a, b] stores a 1
                     each iteration the the column [a, b] is active during the
                     last 1000 iterations.
    :param overlap_sum: the BSP'perm overlapSum matrix (p. 4). This parameter
                        is modified and returned. It is a dictionary with
                        tuples (y, x) of HTM column coordinates as keys and
                        deque (a queue implementation) instances as values. The
                        queue for key [a, b] stores a 1 each time the overlap
                        of the column [a, b] was above the minOverlap threshold
                        during the last 1000 iterations.
    :param min_activity: the BSP'perm minActivity matrix (p. 4); an array
                         min_activity of shape
                         (columns.shape[0], columns.shape[1]), where each
                         min_activity[a b] represents the calculated
                         minActivity for the column [a, b].
    :param boost: the BSP'perm boost matrix (pp. 3, 4).
                  It is a matrix of shape (columns.shape[0], columns.shape[1])
    :param b_inc: the BSP'perm bInc parameter (p. 4). A float that indicates
                  the amount by which a column'perm boost must be incremented.
    :param p_mult: the BSP'perm pMult parameter (p. 4). A float that indicates
                   the amount by which a synapse'perm permanence must be
                   multiplied.
    :param connect_threshold: threshold over which a potential synapse is
                              considered *connected*. All potential synapses
                              start with a permanence value within 0.1 of
                              this parameter.
    :param distances: a 4-dimensional array of the same shape as *columns*;
                      each element distances[a, b, c, d] contains the euclidean
                      distance from (a, b) to (c, d).
    :param b_max: boost threshold (p. 6).
    :return: a tuple (columns, synapse_modified). *columns* is the  parameter
             of the same name, modified according to the BSP learning
             algorithm. *synapse_modified* is a boolean indicating whether any
             synapses were modified in the course of learning.
    """
    # Assume no synapse will be modified.
    synapse_modified = False
    mean_input = np.nanmean(input_vector)

    # Store which synapses are connecter and which aren't.
    pre_col_matrix = ne.evaluate('columns >= connect_threshold')
    # For each active column [y, x] ...
    for _, _, syn_matrix in iter_columns(columns, active_matrix=active):
        # for each potential synapse [u, v] of [y, x] with permanence perm,
        # (NOTE: by definition, perm = syn_matrix[u, v])
        for u, v, perm in iter_synapses(syn_matrix, only_potential=True):
            s = (u, v)
            if (input_vector[s] > mean_input and
                    perm >= connect_threshold):
                syn_matrix[s] = min(perm + p_inc, 1)
            elif (input_vector[s] > mean_input and
                    perm < connect_threshold):
                syn_matrix[s] = min(perm + p_inc,
                                    connect_threshold - p_inc)
            else:
                syn_matrix[s] = max(perm - p_dec, 0)

    # For each column [y, x] ...
    for y, x, syn_matrix in iter_columns(columns):
        c = (y, x)
        # if the activity of [y, x] over the last 1000 iterations was too low,
        if activity[c].sum() < min_activity[c]:
            # increment the boost by b_inc (in the paper this is done inside
            # the if clause in lines 14-15 of algorithm 3 in page 6), ...
            boost[c] += b_inc
            # if its boost is too high, ...
            if boost[c] > b_max:
                # define a function to filter all synapses with a permanence
                # value below the threshold, ...
                def filter_permanences(s):
                    u, v, perm = s
                    if perm < connect_threshold:
                        return (perm, distances[c][u, v])
                    else:
                        return -np.infty

                # reset the boost for column [y, x], ...
                boost[c] = 1
                # select the disconnected synapse with the highest permanence,
                # choosing the nearest synapse in case of tie, ...
                max_syn = max(iter_synapses(syn_matrix),
                              key=filter_permanences)
                max_s = max_syn[:2]
                # and set the selected synapse's permanence value to p_inc
                # above the threshold.
                syn_matrix[max_s] = connect_threshold + p_inc

    # Set synapse_modified to True if any synapse change from connected to
    # disconnected or vice-versa by this algorithm.
    if (pre_col_matrix !=
            ne.evaluate('columns >= connect_threshold')).any():
        synapse_modified = True

    # Return the columns array, with its synapses modified.
    return columns, synapse_modified


def spatial_pooler(images, shape, p_connect=0.15, connect_threshold=0.2,
                   p_inc=0.02, p_dec=0.02, b_inc=0.005, p_mult=0.01,
                   min_activity_threshold=0.01, desired_activity_mult=0.05,
                   b_max=4, max_iterations=10000, cycles_to_save=100,
                   output_file=None):
    """
    Implements the main BSP loop (p. 3). It goes continually through the images
    set until convergence.
    :param images: set of images to learn from. It is an array of shape
                   (l, m, n) where (l, m) == (shape[0], shape[1]) and
                   (l, m) == (shape[2], shape[3])
    :param shape: the shape of the output array. It must have 4 components, and
                  (shape[0], shape[1]) == (shape[2], shape[3]).
    :param p_connect: probability of the columns mapped to [i, j] to have a
                      potential synapse to coordinates [k, l], for all i in
                      shape[0], j in shape[1], k in shape[2], and l in
                      shape[3].
    :param connect_threshold: threshold over which a potential synapse is
                              considered *connected*. All potential synapses
                              start with a permanence value within 0.1 of
                              this parameter.
    :param p_inc: the BSP'perm pInc parameter (p. 4). A float that indicates
                  the amount by which a synapse'perm permanence must be
                  incremented.
    :param p_dec: the BSP'perm pDec parameter (p. 4). A float that indicates
                  the amount by which a synapse'perm permanence must be
                  decremented.
    :param b_inc: the BSP'perm bInc parameter (p. 4). A float that indicates
                  the amount by which a column'perm boost must be incremented.
    :param p_mult: the BSP'perm pMult parameter (p. 4). A float that indicates
                   the amount by which a synapse'perm permanence must be
                   multiplied.
    :param min_activity_threshold: the BSP's minActivityThreshold parameter
                                   (p. 4).
    :param desired_activity_mult: the BSP's desiredActivityMult parameter
                                  (p. 4).
    :param b_max: the ASP'perm bMax parameter (p. 6). A float that indicates
                  the amount by which a synapse'perm permanence must be
                  multiplied.
    :param max_iterations: an integer indicating the maximum number of runs
                           through the set of images allowed. Pass None if no
                           limit is desired.
    :param cycles_to_save: wait this number of iterations over the complete set
                           of images before saving the columns to disk.
    :param output_file: file name used to save the pickled columns.
    :return: a matrix *columns* of shape *shape*, created and modified
             according to the BSP learning algorithm.
    """
    # Initialize boost matrix.
    boost = np.ones(shape=shape[:2])
    part_rbuffer = partial(RingBuffer,
                           input_array=np.zeros(1000, dtype=np.bool),
                           copy=True)
    # Initialize activity dictionary.
    activity = defaultdict(part_rbuffer)
    # Initialize columns and distances matrices.
    pprint("Initializing synapses ...")
    columns, distances = initialise_synapses(shape, p_connect,
                                             connect_threshold)
    pprint("Columns:")
    random_rows = np.random.randint(0, shape[0], 2)
    random_cols = np.random.randint(0, shape[1], 2)
    pprint(columns[random_rows, random_cols])
    pprint("Distances:")
    pprint(distances[random_rows, random_cols])
    # Calculate the inhibition_area parameter.
    pprint("Calculating inhibition area ...")
    inhibition_area = update_inhibition_area(columns, connect_threshold)
    pprint("Inhibition area: %s" % inhibition_area)
    # Calculate the desired activity in a inhibition zone.
    pprint("Calculating desired activity ...")
    desired_activity = desired_activity_mult * inhibition_area
    pprint("Desired activity: %s" % desired_activity)

    converged = False
    i = 0
    # While synapses are modified and the maximum number of iterations is not
    # overstepped, ...
    pprint("Starting learning loop ...")
    start = datetime.now()
    while not converged and (max_iterations is None or i < max_iterations):
        # Initialize the synapses_modified array, assuming no synapses will be
        # modified.
        synapses_modified = np.zeros(shape=len(images), dtype=np.bool)
        # For each image *image*, with index *j* in the images set, ...
        for j, image, _ in read_input(images):
            # According to the paper (sic):
            #   "minOverlap was dynamically set to be the product of the mean
            #    pixel intensity of the current image and the mean number of
            #    connected synapses for an individual column."
            # This leaves unclear exactly what is meant by "mean number of
            # connected synapses for an individual column"; it could be a
            # historical mean or a mean over all columns, here the latter was
            # chosen.
            mean_conn_synapses = (columns[columns > connect_threshold].size /
                                  (shape[2] * shape[3]))
            min_overlap = image.mean() * mean_conn_synapses
            # calculate the overlap of the columns with the image.
            # (this is a simple count of the number of its connected synapses
            # that are receiving active input (p. 3)), ...
            overlap = calculate_overlap(image, columns, min_overlap,
                                        connect_threshold, boost)
            # force sparsity by inhibiting columns, ...
            active, activity =\
                inhibit_columns(columns, distances, inhibition_area,
                                overlap, activity, desired_activity)
            # calculate the min_activity matrix, ...
            min_activity =\
                calculate_min_activity(columns, active, distances,
                                       inhibition_area, activity,
                                       min_activity_threshold)
            # and finally, adapt the synapse's permanence values.
            columns, synapses_modified[j] =\
                learn_synapse_connections(columns, active, image, p_inc,
                                          p_dec, activity, min_activity, boost,
                                          b_inc, p_mult, connect_threshold,
                                          distances, b_max)
            # Update the inhibition_area parameter.
            inhibition_area = update_inhibition_area(columns,
                                                     connect_threshold)
            # Update the desired activity in a inhibition zone.
            desired_activity = desired_activity_mult * inhibition_area
            # Print a snapshot of the model state every 1000 images.
            if j % 1000 == 0:
                pprint("########## %sth image of %sth iteration ##########" %
                       (j+1, i+1))
                elapsed = datetime.now() - start
                elapsed_h = elapsed.total_seconds() // 3600
                elapsed_m = (elapsed.total_seconds() // 60) % 60
                elapsed_s = elapsed.seconds % 60
                pprint("########## Elapsed time: %02d:%02d:%02d ##########" %
                       (elapsed_h, elapsed_m, elapsed_s))
                pprint("Overlap:")
                pprint(overlap[random_rows, random_cols])
                pprint("Activity:")
                for l, key in enumerate(activity.iterkeys()):
                    if l in random_rows:
                        pprint(activity[key][-100:])
                pprint("Active:")
                pprint(active[random_rows, random_cols])
                pprint("Min activity:")
                pprint(min_activity[random_rows, random_cols])
                pprint("Inhibition area: %s" % inhibition_area)
                pprint("Inhibition radius: %s" %
                       (np.sqrt(inhibition_area/np.pi),))
                pprint("Desired activity: %s" % desired_activity)
                pprint("Synapses modified: %s" % synapses_modified[j])
        # Check if any synapses changed from connected to disconnected or
        # vice-versa in the last learning cycle.
        converged = test_for_convergence(synapses_modified)
        pprint("Iteration %s. Number of synapses modified: %s" %
               (i, synapses_modified.sum()))
        if i % cycles_to_save == 0 or converged:
            if output_file is not None:
                with open(output_file, 'wb') as fp:
                    pickle.dump(columns, fp)
        # Increment iterations counter.
        i += 1

    return columns

if __name__ == '__main__':
    import sys
    from utils import load_images, extract_patches

    # Check whether the --output_file command line parameter was provided.
    output_file = '.'
    if '--output_file' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--output_file')
        output_file = sys.argv[arg_index + 1]
    else:
        sys.exit('The parameter --output_file is mandatory.')

    # Check whether the --images_path command line parameter was provided.
    images_path = '.'
    if '--images_path' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--images_path')
        images_path = sys.argv[arg_index + 1]

    # Check whether the --create_patches command line parameter was provided.
    create_patches = False
    if '--create_patches' in sys.argv:
        create_patches = True

    # Check whether the --patches_file command line parameter was provided.
    patches_file = None
    if '--patches_file' in sys.argv:
        # Get the command line parameter value.
        arg_index = sys.argv.index('--patches_file')
        patches_file = sys.argv[arg_index + 1]

    patches = None
    if patches_file is not None:
        # If the patches_file command line parameter was provided,
        # read it from disk.
        pprint("Reading patches file ...")
        with open(patches_file, 'rb') as fp:
            patches = pickle.load(fp)
    else:
        # If the patches_file command line parameter was not provided,
        # load the images from disk, ...
        pprint("Loading images ...")
        images, image_files = load_images(images_path, extensions=['.png'],
                                          img_shape=(256, 256))
        if create_patches:
            # and if the --create_patches was provided generate the patches.
            pprint("Extracting patches ...")
            patches = extract_patches(images, patch_shape=(16, 16))
            if patches_file is not None:
                pprint("Saving patches to disk ...")
                with open(patches_file, 'wb') as fp:
                    pickle.dump(patches, fp)
        else:
            # or, if the --create_patches was not provided, use the images
            # themselves as the patches.
            patches = images

    # Finally, start the learning procedure.
    pprint("Starting training ...")
    columns = spatial_pooler(patches, shape=(16, 16, 16, 16), p_connect=0.1,
                             connect_threshold=0.2, p_inc=0.0005, p_dec=0.0025,
                             b_inc=0.005, p_mult=0.01,
                             min_activity_threshold=0.01,
                             desired_activity_mult=0.05, b_max=4,
                             max_iterations=patches.shape[0],
                             output_file=output_file)
