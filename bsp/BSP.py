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


Implementation of the binary spatial pooler according to:

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

from collections import defaultdict, deque
import cPickle as pickle
from functools import partial
from pprint import pprint

import numpy as np
import numexpr as ne

from utils import (read_input, euclidean_dist, iter_columns,
                   iter_synapses, iter_neighbours)


def initialise_synapses(shape, p_connect, connect_threshold):
    """
    Implements the initializeSynapses function from the paper (p. 3).
    :param shape: the shape of the output array. It must have 4 components.
    :param p_connect: probability of the columns mapped to [i, j] to have a
                      potential synapse to coordinates [k, l], for all i in
                      shape[0], j in shape[1], k in shape[2], and l in
                      shape[3].
    :param connect_threshold: threshold over which a potential synapse is
                              considered *connected*. All potential synapses
                              start with a permanence value within 0.1 of
                              this parameter.
    :return: a tuple (columns, distances). *columns* is a 4-dimensional array
             of shape *shape* that represents the HTM columns and their
             synapses; each element columns[a, b, c, d] contains the permanence
             value of the synapse connecting the column with coordinates (a, b)
             to input with coordinates (c, d). *distances* is also a
             4-dimensional array of shape *shape*; each element
             distances[a, b, c, d] contains the euclidean distance from (a, b)
             to (c, d).
    """
    # Make sure the shape has the right number of dimensions.
    assert len(shape) == 4
    # Make sure the first two sizes are the same as the last two
    # TODO: delete this assert once the mapping from column
    #       coordinates to pixel coordinates is implemented
    assert shape[:2] == shape[2:]
    # Calculate the number of elements the flattened columns array should have.
    # For generating the synapses and permanences, a flatened array is easier
    # to index and work with than the 4-dimensional column matrix. Once the
    # potential synapses and its permanences are calculated, the flattened
    # array will be reshaped to have the final 4-dimensions.
    flattened_size = np.asarray(shape).prod()

    # Create an array of size flattened_size by drawing from a uniform
    # distribution in [0, 1).
    potential_pool = np.random.uniform(size=flattened_size)
    # Make all cells with values under p_connect True and everything else
    # False. The elements that are True are the potential synapses.
    potential_pool = potential_pool <= p_connect

    # Calculate minimum and maximum permanence values.
    min_perm = max(connect_threshold - 0.1, 0)
    max_perm = min(connect_threshold + 0.1, 1)
    # Generate the connected potential synapse's permanence values by
    # drawing from a uniform distribution in [connect_threshold, max_perm).
    connected_perm = np.random.uniform(low=connect_threshold, high=max_perm,
                                       size=flattened_size)
    # Generate the disconnected potential synapse's permanence values by
    # drawing from a uniform distribution in [min_perm, connect_threshold).
    disconnected_perm = np.random.uniform(low=min_perm, high=connect_threshold,
                                          size=flattened_size)

    # Generate the coordinate matrices for the colums and synapses.
    # This will be used to calculate the distance from each HTM column
    # to each input.
    col_coords = np.asarray([[i, j] for i in range(shape[0])
                             for j in range(shape[1])])
    syn_coords = col_coords.copy()

    # Calculate coordinate distances. This will result in a matrix of shape
    # (shape[0]*shape[1], shape[0]*shape[1]).
    distances = np.asarray([euclidean_dist(syn_coords, cc)
                            for cc in col_coords])
    # Flatten the distances array so it can be directly compared with
    # the other arrays.
    distances = distances.flatten()

    # Obtain the biggest value in the distances array.
    max_distance = distances.max()
    # Generate the probabilities of a potential synapse being connected
    # as the inverse of the distance normalized in the range [0, 1],
    # i.e., the synapses nearer to their respective HTM columns will have
    # an associated probaility nearer 1 and vice-versa.
    conn_prob_matrix = 1 - distances/max_distance

    # Draw a random sample of the same size as the flattened column matrix.
    random_sample = np.random.uniform(size=flattened_size)
    # Generate matrices for connected and unconnected *potential* synapses.
    connected_pool = (random_sample <= conn_prob_matrix) & potential_pool
    disconnected_pool = (random_sample > conn_prob_matrix) & potential_pool

    # generate the matrix of columns, filled with NaNs (Not a Number).
    columns = np.array([np.nan]*flattened_size)
    # Assign the permanences of the connected synapses.
    columns[connected_pool] = connected_perm[connected_pool]
    # Assign the permanences of the unconnected synapses.
    columns[disconnected_pool] = disconnected_perm[disconnected_pool]

    # Reshape the distance matrix. The reshaping takes elements from left to
    # right, from last dimension to first dimension.
    # So if, one wants to reshape an array
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # to have shape (2, 2, 2, 2), the result will be an array
    # [
    #   [
    #     [[1, 2], [3, 4]]
    #     [[5, 6], [7, 8]]
    #   ],
    #   [
    #     [[9, 10], [11, 12]]
    #     [[13, 14], [15, 16]]
    #   ]
    # ]
    distances = distances.reshape(shape)
    # Reshape the columns matrix.
    columns = columns.reshape(shape)

    return columns, distances


def calculate_overlap(input_vector, columns, min_overlap, connect_threshold,
                      boost, overlap_sum):
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
        # calculate the overlap as the sum of ON bits in the input_vector
        # assigned to *connected* synapses and, ...
        # (numexpr.evaluate optimizes and carries out the operations defined in
        # the comparison_expr string, it is used here because numpy has some
        # problems when comparing NaNs with numbers)
        input_vector = input_vector.astype(np.bool)
        comparison_expr = '(syn_matrix >= connect_threshold) & input_vector'
        overlap[y, x] = np.sum(ne.evaluate(comparison_expr))
        # if the overlap is not enough, ...
        if overlap[y, x] < min_overlap:
            # reset it, but, ...
            overlap[y, x] = 0
        # if the overlap is enough, ...
        else:
            # first boost it, ...
            overlap[y, x] *= boost[y, x]
            # and then update the overlapSum array, indicating that
            # the column [y, x] was updated in this iteration.
            overlap_sum[y, x].append(1)

    return overlap, overlap_sum


def inhibit_columns(columns, distances, inhibition_area,
                    overlap, activity, desired_activity):
    """
    Implements the inhibitColums function from the paper (pp. 3, 4)
    :param columns: the 4-dimensional array of HTM columns: a 4-dimensional
                    array of shape *shape* that represents the HTM columns and
                    their synapses; each element columns[a, b, c, d] contains
                    the permanence value of the synapse connecting the column
                    with coordinates (a, b) to input with coordinates (c, d).
    :param distances: a 4-dimensional array of the same shape as *columns*;
                      each element distances[a, b, c, d] contains the euclidean
                      distance from (a, b) to (c, d).
    :param inhibition_area: the BSP's inhibitionArea parameter (p. 3, 4).
    :param overlap: an array of shape (columns.shape[0], columns.shape[0]);
                    each element overlap[a, b] contains the overlap of the
                    column [a, b] with the image.
    :param activity: the BSP's activity matrix (p. 4).
                     This parameter is modified and returned.
                     It is a dictionary with tuples (y, x) of HTM column
                     coordinates as keys and deque (a queue implementation)
                     instances as values. The queue for key [a, b] stores a 1
                     each iteration the the column [a, b] is active during the
                     last 1000 iterations.
    :param desired_activity: the BSP's desiredActivity parameter (p. 4).
    :return: a tuple (active, activity). *active* is an array of shape
             (columns.shape[0], columns.shape[0]); each element active[a, b] is
             True if the column [a, b] is active this iteration and False
             otherwise. *activity* is the parameter of the same name; the queue
             in activity[a, b] will have a 1 pushed into it if the the column
             [a, b] was active this iteration.
    """
    # Initialize the active array filling it with False.
    active = np.zeros(shape=columns.shape[:2], dtype=np.bool)
    # Calculate the inhibition radius. It is easier to check if the distance
    # from column a to column b is less than the inhibition radius, than to
    # check whether column b lies inside column a's inhibition area.
    # Both approaches are equivalent, since the inhibition area is a circle.
    inhibition_radius = 4 * np.sqrt(inhibition_area/np.pi)

    # For each column [y, x]...
    for y, x, _ in iter_columns(columns):
        # if [y, x] reacted to the input at all, ...
        # (NOTE: This test is not present in the ASP paper, but it is present
        # in the original HTM paper. However it does make sense that a column
        # should be active only if some synapses were exited by the input.)
        if overlap[y, x] > 0:
            # initialize the activity counter, ...
            activity_sum = 0
            # obtain the list of neighbours of the column [y, x], ...
            neighbours = iter_neighbours(columns, y, x, distances,
                                         inhibition_radius)
            # for each neighbour [u, v] of [y, x] ...
            for u, v, _ in neighbours:
                # if the neighbour's overlap is over this column's overlap, ...
                if overlap[u, v] > overlap[y, x]:
                    # count it towards the activity, ...
                    activity_sum += 1

            # and if the neighbours of [y, x] are not too active, ...
            if activity_sum < desired_activity:
                # set [y, x] as active, ...
                active[y, x] = True
                # and then update the activity array, indicating that
                # the column [y, x] was active in this iteration.
                activity[y, x].append(1)

    return active, activity


def calculate_inhibition_area(columns):
    """
    Implements the updateInhibitionArea function from the paper (p. 4). The
    name was changed to reflect that the updateInhibitionArea can be calculated
    just once, since the size of the inhibition area does not change during
    learning.
    :param columns: the 4-dimensional array of HTM columns: a 4-dimensional
                    array of shape *shape* that represents the HTM columns and
                    their synapses; each element columns[a, b, c, d] contains
                    the permanence value of the synapse connecting the column
                    with coordinates (a, b) to input with coordinates (c, d).
    :return: the inhibition area for the algorithm, calculated as the mean size
             of the receptive fields of all columns, where a column's receptive
             field is calculated as \pi*d^2/16, with:
                d = max([a, b], x) - min([a, b], x) +
                    max([a, b], y) - min([a, b], y) + 2
             where max and min return the the maximum and minimum x and y
             values of all synapses belonging to column [a, b].
    """
    # Initialize the inhibition area accumulator.
    inhibition_area = 0

    # For each column ...
    for _, _, syn_matrix in iter_columns(columns):
        # set d = max([a, b], y) - min([a, b], y), ...
        d = syn_matrix.argmax(axis=0).max() - syn_matrix.argmin(axis=0).max()
        # add max([a, b], x) - min([a, b], x) to d, ...
        d += syn_matrix.argmax(axis=1).max() - syn_matrix.argmin(axis=1).max()
        # add 2 to d, ...
        d += 2
        # calculate the receptive field based on d, ...
        receptive_radius = d/4
        receptive_field = np.pi * receptive_radius**2
        # add the receptive field of the column [y, x] to the accumulator, ...
        inhibition_area += receptive_field

    # and finally calculate the average of all receptive fields.
    inhibition_area /= columns.shape[0]*columns.shape[1]

    return inhibition_area


def calculate_min_activity(columns, active, distances, inhibition_area,
                           activity, min_activity_threshold):
    """
    Calculates the minActivity matrix from the paper (p. 4).
    :param columns: the 4-dimensional array of HTM columns: a 4-dimensional
                    array of shape *shape* that represents the HTM columns and
                    their synapses; each element columns[a, b, c, d] contains
                    the permanence value of the synapse connecting the column
                    with coordinates (a, b) to input with coordinates (c, d).
    :param active: array of shape (columns.shape[0], columns.shape[0]);
                   each element active[a, b] is True if the column [a, b]
                   is active this iteration and False otherwise.
    :param distances: a 4-dimensional array of the same shape as *columns*;
                      each element distances[a, b, c, d] contains the euclidean
                      distance from (a, b) to (c, d).
    :param inhibition_area: the BSP's inhibitionArea parameter (p. 3, 4).
    :param activity: the BSP's activity matrix (p. 4).
                     This parameter is modified and returned.
                     It is a dictionary with tuples (y, x) of HTM column
                     coordinates as keys and deque (a queue implementation)
                     instances as values. The queue for key [a, b] stores a 1
                     each iteration the the column [a, b] is active during the
                     last 1000 iterations.
    :param min_activity_threshold: the BSP's minActivityThreshold parameter
                                   (p. 4).
    :return: the BSP's minActivity matrix (p. 4); an array min_activity of
             shape (columns.shape[0], columns.shape[1]), where each
             min_activity[a b] represents the calculated minActivity for the
             column [a, b]:

             minActivity([a, b]) = max(activity[u, v]) * minActivityThreshold

             where the column [u, v] is a neighbour of [y, x] (within
             inhibitionArea)
    """
    # Initialize the min_activity array.
    min_activity = np.zeros(shape=columns.shape[:2])
    # Calculate the inhibition radius. It is easier to check if the distance
    # from column a to column b is less than the inhibition radius, than to
    # check whether column b lies inside column a's inhibition area.
    # Both approaches are equivalent, since the inhibition area is a circle.
    inhibition_radius = 4 * np.sqrt(inhibition_area/np.pi)

    # For each column [y, x], ...
    for y, x, _ in iter_columns(columns):
        # if [y, x] is active this itearation, ...
        if active[y, x]:
            max_activity = 0
            # get the neighbours of [y, x] ...
            neighbours = iter_neighbours(columns, y, x, distances,
                                         inhibition_radius)
            # and for each neighbour [u, v] of [y, x], ...
            for u, v, _ in neighbours:
                # calculate the how many times [u, v] was active during the
                # last 1000 iterations, ...
                activity_count = sum(activity[u, v])
                # and choose the maximum count among all the neighbours of
                # [y, x].
                if activity_count > max_activity:
                    max_activity = activity_count
            # Finally, scale the maximum activity count among the neighbours of
            # [y, x] by min_activity_threshold.
            min_activity[y, x] = max_activity * min_activity_threshold

    return min_activity


def learn_synapse_connections(columns, active, input_vector, p_inc,
                              p_dec, activity, overlap_sum,
                              min_activity, boost, b_inc, p_mult):
    """
    Calculates the minActivity matrix from the paper (p. 4).
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
    :return: a tuple (columns, synapse_modified). *columns* is the  parameter
             of the same name, modified according to the BSP learning
             algorithm. *synapse_modified* is a boolean indicating whether any
             synapses were modified in the course of learning.
    """
    # Assume no synapse will be modified.
    synapse_modified = False
    # For each column [y, x] ...
    for y, x, syn_matrix in iter_columns(columns):
        # if [y, x] is active in this iteration, ...
        if active[y, x]:
            # for each synapse [u, v] of [y, x] with permanence perm, ...
            # (NOTE: by definition, perm = syn_matrix[u, v])
            for u, v, perm in iter_synapses(syn_matrix):
                # increase perm, truncated at 1, if input_vector[u, v]==1, ...
                if input_vector[u, v]:
                    syn_matrix[u, v] = min(perm + p_inc, 1)
                # or decrease perm, truncated at 0, if input_vector[u, v] == 0.
                else:
                    syn_matrix[u, v] = max(perm - p_dec, 0)
            # Set synapse_modified to True if any synapse was modified by this
            # algorithm.
            if syn_matrix[u, v] != perm:
                synapse_modified = True

    # For each column [y, x] ...
    for y, x, syn_matrix in iter_columns(columns):
        # increment the boost for [y, x] if the activity of [y, x] is too low,
        if sum(activity[y, x]) < min_activity[y, x]:
            boost[y, x] += b_inc
        # or reset it to 1 otherwise.
        else:
            boost[y, x] = 1

        # Finally, if the amount of times [y, x] was over the overlapThreshold
        # in the last 1000 iterations is below the minimum activity, ...
        if sum(overlap_sum[y, x]) < min_activity[y, x]:
            # for each neighbour [u, v] of [y, x], ...
            # (NOTE: by definition, perm = syn_matrix[u, v])
            for u, v, perm in iter_synapses(syn_matrix):
                # multiply perm by p_mult, truncated at 1.
                syn_matrix[u, v] = min(perm * p_mult, 1)
                # Set synapse_modified to True if any synapse was modified by
                # this algorithm.
                if syn_matrix[u, v] != perm:
                    synapse_modified = True

    # Return the columns array, with its synapses modified.
    return columns, synapse_modified


def test_for_convergence(synapses_modified):
    """
    Evaluates whether the algorithm has converged.
    :param synapses_modified: a 1-dimensional array of length len(images). Each
                             element synapses_modified[i] is True if a synapse
                             was modified while learning images[i].
    :return: True if no synapses were modified in the last run through all the
             images, and False otherwise.
    """
    return not synapses_modified.any()


def spatial_pooler(images, shape, p_connect=0.15, connect_threshold=0.2,
                   p_inc=0.02, p_dec=0.02, b_inc=0.005, p_mult=0.01,
                   min_activity_threshold=0.01, min_overlap=3,
                   desired_activity_mult=0.05, max_iterations=10000,
                   cycles_to_save=100, output_file=None):
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
    :param min_overlap: the BSP's minOverlap parameter (p. 3). Type: float.
    :param desired_activity_mult: the BSP's desiredActivityMult parameter
                                  (p. 4).
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
    # Initialize overlap_sum dictionary.
    part_deque = partial(deque, maxlen=1000)
    overlap_sum = defaultdict(part_deque)
    # Initialize activity dictionary.
    activity = defaultdict(part_deque)

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
    inhibition_area = calculate_inhibition_area(columns)
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
    while not converged and (max_iterations is None or i < max_iterations):
        # Initialize the synapses_modified array, assuming no synapses will be
        # modified.
        synapses_modified = np.zeros(shape=len(images), dtype=np.bool)
        # For each image *image*, with index *j* in the images set, ...
        for j, image, _ in read_input(images):
            # calculate the overlap of the columns with the image.
            # (this is a simple count of the number of its connected synapses
            # that are receiving active input (p. 3)), ...
            overlap, overlap_sum =\
                calculate_overlap(image, columns, min_overlap,
                                  connect_threshold, boost, overlap_sum)
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
                                          p_dec, activity, overlap_sum,
                                          min_activity, boost, b_inc, p_mult)
            # Print a snapshot of the model state every 1000 images.
            if j % 1000 == 0:
                pprint("########## %sth iteration ##########" % j)
                pprint("Overlap:")
                pprint(overlap[random_rows, random_cols])
                pprint("Overlap sum:")
                for l, key in enumerate(overlap_sum.iterkeys()):
                    if l in random_rows:
                        pprint(overlap_sum[key])
                pprint("Activity:")
                for l, key in enumerate(activity.iterkeys()):
                    if l in random_rows:
                        pprint(activity[key])
                pprint("Active:")
                pprint(active[random_rows, random_cols])
                pprint("Min activity:")
                pprint(min_activity[random_rows, random_cols])
                pprint("Synapses modified:")
                pprint(synapses_modified[j])
        # Check if any synapses were modified in the last learning cycle.
        converged = test_for_convergence(synapses_modified)
        pprint("Iteration %s. Columns modified: %s" %
               (i, synapses_modified.sum()))
        if output_file is not None and (i % cycles_to_save == 0 or converged):
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
    columns = spatial_pooler(patches, shape=(16, 16, 16, 16), p_connect=0.15,
                             connect_threshold=0.2,
                             p_inc=0.02, p_dec=0.02, b_inc=0.005, p_mult=0.01,
                             min_activity_threshold=0.01, min_overlap=3,
                             desired_activity_mult=0.05,
                             output_file=output_file)
    with open(output_file, 'wb') as fp:
        pickle.dump(columns, fp)
