
import os
import os.path as path

from PIL import Image
import numpy as np


def extract_patches(images, patch_shape, patches_nr=None):
    img_shape = images.shape[1:]
    x_high = img_shape[1] - patch_shape[1] + 1
    y_high = img_shape[0] - patch_shape[0] + 1
    if patches_nr is None:
        patches_per_img = img_shape[0] - patch_shape[0]
        patches_per_img *= img_shape[1] - patch_shape[1] + 1
        patches_nr = patches_per_img * images.shape[0]
    patches = np.empty((patches_nr, patch_shape[0], patch_shape[1]))

    for i in xrange(patches_nr):
        img_index = np.random.randint(low=0, high=images.shape[0])
        y_from = np.random.randint(low=0, high=y_high)
        y_to = y_from + patch_shape[0]
        x_from = np.random.randint(low=0, high=x_high)
        x_to = x_from + patch_shape[1]
        patches[i] = images[img_index, y_from:y_to, x_from:x_to]

    return patches


def load_images(images_path, extensions=('.png',), img_shape=(256, 256)):
    """
    Loads images from images_path into a np.array.
    :param images_path: path to the directory containing the images. The search is not recursive.
                        The images will be converted to grayscale mode w/o alpha channel.
    :param extensions: file extensions to be returned by the search. The extensions must start with a '.'
                       and be lowercase.
    :param images_path: shape of the images. All images found will be reshaped according to this parameter.
    :return: a tuple (np.array of shape (len(image files), img_shape[0], img_shape[1]), list of file paths)
    """
    # traverse images_path directory and get all files matching the extensions
    image_files = [path.join(images_path, f) for f in os.listdir(images_path)
                   if (path.isfile(path.join(images_path, f)) and
                       path.splitext(f)[1].lower() in extensions)]
    # create the return array
    images = np.empty((len(image_files), img_shape[0], img_shape[1]))
    # load each file found, ...
    for i, image_file in enumerate(image_files):
        img = Image.open(image_file).convert('L')
        # resize it to the correct dimensions ...
        img = img.resize(img_shape)
        # and add it to the return array
        images[i] = np.asarray(img)
    return images, image_files


def read_input(images):
    """
    Randomly yields an image without replacement from **images**.
    :param images: np.array of images
    :return: an iterator. Yields a tuple (np.array with an image, index of the image in **images**)
    """
    # create an index array
    images_order = np.arange(len(images))
    # shuffle the index
    np.random.shuffle(images_order)
    # iterate over the shuffled index
    for i, image_index in enumerate(images_order):
        yield i, images[image_index], image_index


def euclidean_dist(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    dist = (v1 - v2)**2
    dist = np.sum(dist, axis=1)
    return np.sqrt(dist)
