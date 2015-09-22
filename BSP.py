"""
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

from collections import defaultdict, deque

import numpy as np

from utils import read_input, euclidean_dist


def iter_columns(columns):
    for x, col_row in enumerate(columns):
        for y, syn_matrix in enumerate(col_row):
            yield x, y, syn_matrix


def iter_synapses(synapses):
    for x, syn_row in enumerate(synapses):
        for y, synapse in enumerate(syn_row):
            yield x, y, synapse


def iter_neighbours(columns, x, y, distances, inhibition_area):
    neighbours = columns[distances[x, y] <= inhibition_area]
    for u, v, syn_matrix in iter_columns(columns):
        if neighbours[u, v]:
            yield u, v, syn_matrix


def initialise_synapses(shape, p_connect, connect_threshold):
    """
    :param shape: the shape of the output array. It must have 4 components.
    :param p_connect: probability of the columns mapped to [i, j] to have a
                      potential synapse to coordinates [k, l], for all i in
                      shape[0], j in shape[1], k in shape[2], and l in shape[3].
    :param connect_threshold: threshold over which a potential synapse is
                              considered *connected*. All potential synapses
                              start with a permanence value within 0.1 of
                              this parameter.
    :return:
    """
    # make sure the shape has the right number of dimensions.
    assert len(shape) == 4
    # make sure the first two sizes are the same as the last two
    assert shape[:2] == shape[2:]
    # calculate the number of elements the flattened columns array should have.
    flattened_size = np.asarray(shape).prod()

    # generate the columns array by drawing from a uniform distribution in [0, 1).
    potential_pool = np.random.uniform(size=flattened_size)
    # make all cells with values under p_connect True and everything else False.
    potential_pool = potential_pool <= p_connect

    # calculate minimum and maximum permanence values
    min_perm = connect_threshold - 0.1
    max_perm = connect_threshold + 0.1
    # generate the connected synapse permanence values by drawing from a uniform
    # distribution in [connect_threshold, max_perm).
    connected_perm = np.random.uniform(low=connect_threshold, high=max_perm,
                                       size=flattened_size)
    # generate the unconnected synapse permanence values by drawing from a uniform
    # distribution in [min_perm, connect_threshold).
    unconnected_perm = np.random.uniform(low=min_perm, high=connect_threshold,
                                         size=flattened_size)

    # generate the coordinate matrices for the colums and synapses
    col_coords = np.asarray([[i, j] for i in range(shape[0])
                             for j in range(shape[1])])
    syn_coords = col_coords.copy()

    # calculate coordinate distances. This will result in a matrix of shape
    # (shape[0]*shape[1], shape[0]*shape[1]).
    distances = np.asarray([euclidean_dist(syn_coords, cc)
                            for cc in col_coords])
    # reshape the distance matrix
    distances = distances.reshape(shape)

    # calculate the probability of a synapse being connected based on
    # the inverse of the distance to its column
    max_distance = distances.max()
    conn_prob = 1 - distances/max_distance
    random_sample = np.random.rand(*shape)
    # generate matrices for connected and unconnected *potential* synapses
    connected_pool = (random_sample <= conn_prob) & potential_pool
    unconnected_pool = (random_sample > conn_prob) & potential_pool

    # generate the matrix of columns, filled with NaNs (Not a Number).
    columns = np.array([np.nan]*flattened_size).reshape(shape)
    # assign the permanences of the connected synapses
    columns[connected_pool] = connected_perm[connected_pool]
    # assign the permanences of the unconnected synapses
    columns[unconnected_pool] = unconnected_perm[unconnected_pool]

    return columns, distances


def calculate_overlap(image, columns, min_overlap, connect_threshold,
                      boost, overlap_sum):
    overlap = np.zeros(columns.shape[:2])
    for x, y, syn_matrix in iter_columns(columns):
        overlap[x, y] = (syn_matrix >= connect_threshold) & image
        if overlap[x, y] < min_overlap:
            overlap[x, y] = 0
        else:
            overlap_sum[x, y].append(1)
            overlap[x, y] *= boost[x, y]
    return overlap, overlap_sum


def inhibit_columns(columns, distances, inhibition_area,
                    overlap, activity, desired_activity):

    active = np.zeros(shape=columns.shape[:2])

    for x, y, syn_matrix in iter_columns(columns):
        active[x, y] = False
        activity_sum = 0
        neighbours = iter_neighbours(columns, x, y, distances,
                                     inhibition_area)
        for u, v, _ in neighbours:
            if overlap[u, v] > overlap[x, y]:
                activity_sum += 1

        if activity_sum < desired_activity:
            active[x, y] = True
            activity[x, y].append(1)

    return active, activity


def calculate_inhibition_area(columns):
    inhibition_area = 0

    for x, y, syn_matrix in iter_columns(columns):
        d = syn_matrix.argmax() - syn_matrix.argmin()
        d += syn_matrix.argmax(axis=1) - syn_matrix.argmin(axis=1)
        d += 2
        receptive_field = np.pi * d**2 / 16
        inhibition_area += receptive_field

    inhibition_area /= columns.shape[0]*columns.shape[1]

    return inhibition_area


def calculate_min_activity(columns, active, distances, inhibition_area,
                           activity, min_activity_threshold):
    min_activity = np.zeros(shape=columns.shape[:2])
    for x, y, syn_matrix in iter_columns(columns):
        if active[x, y]:
            max_activity = 0
            neighbours = iter_neighbours(columns, x, y, distances,
                                         inhibition_area)
            for u, v, _ in neighbours:
                if activity[u, v] > max_activity:
                    max_activity = activity[u, v]

            min_activity[x, y] = max_activity * min_activity_threshold

    return min_activity


def learn_synapse_connections(columns, active, input_vector, p_inc,
                              p_dec, activity, overlap_sum,
                              min_activity, boost, b_inc, p_mult):
    for x, y, syn_matrix in iter_columns(columns):
        if active[x, y]:
            for u, v, s in iter_synapses(syn_matrix):
                if input_vector[u, v]:
                    syn_matrix[u, v] = min(s + p_inc, 1)
                else:
                    syn_matrix[u, v] = max(s - p_dec, 0)

    synapse_modified = False
    for x, y, syn_matrix in iter_columns(columns):
        if sum(activity[x, y]) < min_activity[x, y]:
            boost[x, y] += b_inc
        else:
            boost[x, y] = 1

        if overlap_sum[x, y] < min_activity[x, y]:
            for u, v, s in iter_synapses(syn_matrix):
                syn_matrix[u, v] = min(s * p_mult, 1)
                if syn_matrix[u, v] != s:
                    synapse_modified = True

    return columns, synapse_modified


def test_for_convergence(synapse_modified):
    return not synapse_modified.any()


def spatial_pooler(images, shape, p_connect=0.15, connect_threshold=0.2,
                   p_inc=0.02, p_dec=0.02, b_inc=0.005, p_mult=0.01,
                   min_activity_threshold=0.01, min_overlap=3,
                   desired_activity_mult=0.05, d=100):
    boost = np.ones(shape=shape[:2])
    overlap_sum = defaultdict(deque(maxlen=1000))
    activity = defaultdict(deque(maxlen=1000))

    columns, distances = initialise_synapses(shape, p_connect,
                                             connect_threshold)
    inhibition_area = calculate_inhibition_area(columns)

    converged = False
    while not converged:
        synapse_modified = np.zeros(shape=len(images), dtype=np.bool)
        for i, image, _ in read_input(images):
            overlap, overlap_sum = calculate_overlap(image, columns, min_overlap,
                                                     connect_threshold, boost,
                                                     overlap_sum)
            desired_activity = desired_activity_mult * inhibition_area
            active, activity = inhibit_columns(columns, distances, inhibition_area,
                                               overlap, activity, desired_activity)
            min_activity = calculate_min_activity(columns, active,
                                                  distances, inhibition_area,
                                                  activity, min_activity_threshold)
            columns, synapse_modified[i] = \
                learn_synapse_connections(columns, active, image, p_inc,
                                          p_dec, activity, overlap_sum,
                                          min_activity, boost, b_inc, p_mult)

        converged = test_for_convergence(synapse_modified)
    return columns


if __name__ == '__main__':
    import sys
    from utils import load_images, extract_patches
    import cPickle as pickle

    images_path = '.'
    if '--images_path' in sys.argv:
        arg_index = sys.argv.index('--images_path')
        images_path = sys.argv[arg_index + 1]

    create_patches = False
    if '--create_patches' in sys.argv:
        create_patches = True

    patches_file = None
    if '--patches_file' in sys.argv:
        arg_index = sys.argv.index('--patches_file')
        patches_file = sys.argv[arg_index + 1]

    patches = None
    if patches_file is not None:
        with open(patches_file, 'rb') as fp:
            patches = pickle.load(fp)
    else:
        images, image_files = load_images(images_path, extensions=['.png'],
                                          img_shape=(256, 256))
        if create_patches:
            patches = extract_patches(images, patch_shape=(16, 16))
            if patches_file is not None:
                with open(patches_file, 'wb') as fp:
                    pickle.dump(patches, fp)
        else:
            patches = images

    spatial_pooler(patches, shape=(16, 16, 16, 16), p_connect=0.15,
                   connect_threshold=0.2,
                   p_inc=0.02, p_dec=0.02, b_inc=0.005, p_mult=0.01,
                   min_activity_threshold=0.01, min_overlap=3,
                   desired_activity_mult=0.05, d=100)
