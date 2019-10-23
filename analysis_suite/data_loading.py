"""
The data_loading.py file for galeria imaging

Contains functions that load data
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>
import os
import skimage.io as skio
import analysis_suite.BR_reader.reader as biorad_reader

def load_image(filepath):
    """
    Takes a filepath and loads the image

    Parameters
    ------
    filepath : str
        Path to the image file

    Returns
    ------
    img : ndarray
        The image
    """
    if os.path.isfile(filepath):
        img = biorad_reader.Reader(filepath)
        loaded_image = img.get_image()
    return loaded_image
