# SpatialPooler
This has spatial pooler code

# Structure:

The *spatialpooler* package contains all the code.

The *spatialpooler/BSP.py* file contains the Binary Spatial Pooler
implementation.

The *spatialpooler/ASP.py* file contains the Augmented Spatial Pooler
implementation.

The *spatialpooler/common.py* file contains all functions common to both the
BSP and ASP.

The *spatialpooler/utils.py* file contains all ancillary functions.

The *spatialpooler/evaluate_experiments.py* file contains the code use for the
pooler's evaluation.

The spatialpooler/test package contains all test files.

Finally, the *images* directory contains the images that where used for
training.

# Prerequisites:

The BSP and ASP implementations depend on numpy, numexpr and matplotlib.
The three can be installed using the Anaconda distribution by:

* Downloading the installer appropriate for your OS from
  [the Continuum Analytics' website](https://www.continuum.io/downloads).
* Installing Anaconda.
* Opening a terminal and typing:
&nbsp;

    conda install numpy numexpr matplotlib


# Use:

The BSP accepts boolean numpy arrays only. The ASP handles single channel
(grayscale) images.

To perform the experiments the RGB images where transformed to grayscale and,
in the case of the BSP, to boolean by setting the pixels with luminosity above
the mean to 1 and below the mean to 0.

## Binary Spatial Pooler:

In order to run the BSP, you can use the following command (from the
repository's root):

    spatialpooler/BSP.py --output_file ./columns2k-8img_bsp.pickle --patches_file patches2k-8img_binary.pickle

columns2k-8img_bsp.pickle is the name of the file where (pickled) trained
columns, in the form of a 4D numpy array, will be stored.

patches2k-8img_binary.pickle is a pre-generated patches taken from some of the
images in the *images* directory. It is a pickled numpy array of shape
(2000, 16, 16) and type numpy.bool, generated using 8 images.

## Augmented Spatial Pooler:

In order to run the ASP, you can use the following command (from the
repository's root):

    spatialpooler/ASP.py --output_file ./columns2k-8img_asp.pickle --patches_file patches2k-8img_grayscale.pickle


columns2k-8img_asp.pickle is the name of the file where (pickled) trained
columns, in the form of a 4D numpy array, will be stored.
patches2k-8img_grayscale.pickle is a pre-generated patches taken from some of
the images in the *images* directory. It is a pickled numpy array of shape
(2000, 16, 16) and type numpy.int, generated using 8 images.
