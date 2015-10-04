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

import numpy as np
import numexpr as ne

from utils import euclidean_dist, iter_columns, iter_neighbours


def calculate_distances(shape):
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

    return distances


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

    distances = calculate_distances(shape)
    # Obtain the biggest value in the distances array.
    max_distance = distances.max()
    # Generate the probabilities of a potential synapse being connected
    # as the inverse of the distance normalized in the range [0, 1],
    # i.e., the synapses nearer to their respective HTM columns will have
    # an associated probaility nearer 1 and vice-versa.
    conn_prob_matrix = 1 - distances.flatten()/max_distance

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

    # Reshape the columns matrix.
    columns = columns.reshape(shape)

    return columns, distances


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
                                         inhibition_area)
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
            else:
                # and then update the activity array, indicating that
                # the column [y, x] was inactive in this iteration.
                activity[y, x].append(0)
        else:
            # update the activity array, indicating that
            # the column [y, x] was inactive in this iteration.
            activity[y, x].append(0)

    return active, activity


def create_min_max_matrices(coord_matrix, syn_matrix, connect_threshold):
    """
    :param coord_matrix: matrix of synapse's coordinates. Each element [i, j]
                         of coord_matrix should be equal to i+j. For instane,
                         for a synapse's matrix of shape (5, 5), coord_matrix
                         should be:
                         [[0, 1, 2, 3, 4],
                         [1, 2, 3, 4, 5],
                         [2, 3, 4, 5, 6],
                         [3, 4, 5, 6, 7],
                         [4, 5, 6, 7, 8]]
    :param syn_matrix: the matrix of synapse's permanences.
    :param connect_threshold: the BSP's connectThreshold parameter (p. 3).
                              Type: float.
    :return: a tuple (min_matrix, max_matrix) where the elements of min_matrix
             are equal to coord_matrix wherever syn_matrix >= connect_threshold
             and equal to +infinity everywhere else. The elements in max_matrix
             are equal to coord_matrix wherever syn_matrix >= connect_threshold
             and equal to -infinity everywhere else.
    """
    # (The elements in min_matrix will be equal to coord_matrix wherever
    # syn_matrix >= connect_threshold and equal to +infinity everywhere
    # else. This matrix will be used to calculate the minimum connected
    # synapse's coordinates)
    min_matrix = (coord_matrix *
                  ne.evaluate('syn_matrix >= connect_threshold'))
    # set all 0 and np.nan elements to +infinity
    min_matrix[(min_matrix == 0) | np.isnan(min_matrix)] = np.infty

    # (The elements in max_matrix will be equal to coord_matrix wherever
    # syn_matrix >= connect_threshold and equal to -infinity everywhere
    # else. This matrix will be used to calculate the maximum connected
    # synapse's coordinates)
    max_matrix = (coord_matrix *
                  ne.evaluate('syn_matrix >= connect_threshold'))
    # set all 0 and np.nan elements to -infinity
    max_matrix[(min_matrix == 0) | np.isnan(min_matrix)] = -np.infty
    return min_matrix, max_matrix


def calculate_min_max_y_x(min_matrix, max_matrix):
    """
    :param min_matrix: The elements of min_matrix should be equal to
                       coord_matrix wherever syn_matrix >= connect_threshold
                       and equal to +infinity everywhere else.
    :param max_matrix: The elements in max_matrix should be equal to
                       coord_matrix wherever syn_matrix >= connect_threshold
                       and equal to -infinity everywhere else.
    :return: a tuple (min_y, min_x, max_y, max_x), where min_y is the minimum y
             coordinate among all connected synapses, min_y is the minimum x
             coordinate among all connected synapses, max_y is the maximum y
             coordinate among all connected synapses, and max_y is the maximum
             x coordinate among all connected synapses.
    """
    # If any of min_matrix or max_matrix are all infinity or -infinity,
    # respectively, then no element was above the connection threshold, so
    # return 0 for all coordinates.
    if (min_matrix == np.infty).all() or (max_matrix == -np.infty).all():
        return 0, 0, 0, 0

    # For each column, get the index of the minimum element of that column, ...
    min_y = min_matrix.argmin(axis=0)
    # then create a filter eliminating all columns with only infinity elements,
    min_y_valid = (min_matrix != np.infty).any(axis=0)
    # and finally apply the filter and get the index of minimum element among
    # all valid columns.
    min_y = min_y[min_y_valid].min()

    # For each row, get the index of the minimum element of that row, ...
    min_x = min_matrix.argmin(axis=1)
    # then create a filter eliminating all rows with only infinity elements,
    min_x_valid = (min_matrix != np.infty).any(axis=1)
    # and finally apply the filter and get the index of minimum element among
    # all valid rows.
    min_x = min_x[min_x_valid].min()

    # For each column, get the index of the maximum element of that column, ...
    max_y = max_matrix.argmax(axis=0)
    # then create a filter eliminating all columns with only -infinity
    # elements,
    max_y_valid = (max_matrix != -np.infty).any(axis=0)
    # and finally apply the filter and get the index of maximum element among
    # all valid columns.
    max_y = max_y[max_y_valid].max()

    # For each row, get the index of the maximum element of that row, ...
    max_x = max_matrix.argmax(axis=1)
    # then create a filter eliminating all rows with only -infinity elements,
    max_x_valid = (max_matrix != -np.infty).any(axis=1)
    # and finally apply the filter and get the index of maximum element among
    # all valid rows.
    max_x = max_x[max_x_valid].max()

    return min_y, min_x, max_y, max_x


def update_inhibition_area(columns, connect_threshold):
    """
    Implements the updateInhibitionArea function from the paper (p. 4).
    :param columns: the 4-dimensional array of HTM columns: a 4-dimensional
                    array of shape *shape* that represents the HTM columns and
                    their synapses; each element columns[a, b, c, d] contains
                    the permanence value of the synapse connecting the column
                    with coordinates (a, b) to input with coordinates (c, d).
    :param connect_threshold: threshold over which a potential synapse is
                              considered *connected*. All potential synapses
                              start with a permanence value within 0.1 of
                              this parameter.
    :return: the inhibition area for the algorithm, calculated as the mean size
             of the receptive fields of all columns, where a column's receptive
             field is calculated as \pi*d^2/16, with:
                d = max([a, b], x) - min([a, b], x) +
                    max([a, b], y) - min([a, b], y) + 2
             where max and min return the the maximum and minimum x and y
             values of all connected synapses belonging to column [a, b].
    """
    # Initialize the inhibition area accumulator.
    inhibition_area = 0
    # Get the shape of the synapse's matrix
    shape = columns.shape[2:]
    # Generate matrix of synapse's coordinates. For instane, for a synapse's
    # matrix of shape (5, 5), coord_matrix would be:
    # [[0, 1, 2, 3, 4],
    #  [1, 2, 3, 4, 5],
    #  [2, 3, 4, 5, 6],
    #  [3, 4, 5, 6, 7],
    #  [4, 5, 6, 7, 8]]
    coord_matrix = np.array([i+j for i in range(shape[0])
                             for j in range(shape[1])], dtype=np.float)
    coord_matrix = coord_matrix.reshape(shape)
    # For each column ...
    for _, _, syn_matrix in iter_columns(columns):
        # create matrices to calculate the synapse's min and max coordinates
        min_matrix, max_matrix =\
            create_min_max_matrices(coord_matrix, syn_matrix,
                                    connect_threshold)
        # calculate the synapse's min and max coordinates
        min_y, min_x, max_y, max_x =\
            calculate_min_max_y_x(min_matrix, max_matrix)
        # set d = max([a, b], y) - min([a, b], y)
        #         + max([a, b], x) - min([a, b], x)
        #         + 2, ...
        d = max_y - min_y + max_x - min_x + 2
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

    # For each active column [y, x], ...
    for y, x, _ in iter_columns(columns, active_matrix=active):
        c = (y, x)
        max_activity = 0
        # get the neighbours of [y, x] ...
        neighbours = iter_neighbours(columns, y, x, distances,
                                     inhibition_area)
        # and for each neighbour [u, v] of [y, x], ...
        for u, v, _ in neighbours:
            n = (u, v)
            # calculate the how many times [u, v] was active during the
            # last 1000 iterations, ...
            activity_count = activity[n].sum()
            # and choose the maximum count among all the neighbours of
            # [y, x].
            if activity_count > max_activity:
                max_activity = activity_count
        # Finally, scale the maximum activity count among the neighbours of
        # [y, x] by min_activity_threshold.
        min_activity[c] = max_activity * min_activity_threshold

    return min_activity


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
