"""
The measurements.py file for galleria_imaging

Contains the main functions for extracting measurements
"""
import numpy as np
from skimage.measure import regionprops

def extract_biolum_values(labelled_wells, fluo_image):
    """
    # TODO:  this function shouldn't stay in detection
    Extracts bioluminescence values from the fluo image by over laying the detected plate

    Parameters
    ------
    labelled_wells : labelled image
        ndi labelled image from skimage.ndimage - where each well has a unique number starting from 1
    fluo_image : array
        Array of the image containing the fluorescence information

    Returns
    ------
    bioluminescence_dict : dictionary
        Where key is the region label and value is a list of measurements [area, fluo, integarted fluo]
    """

    fluo_image = np.array(fluo_image)

    bioluminescence_dict = {}

    for region in regionprops(labelled_wells):
        # get mean fluo from the well area in matching fluo image
        fluo = np.mean(fluo_image[tuple(np.array(region.coords).T)])
        bioluminescence_dict[region.label] = [region.area, fluo, fluo*region.area]

    return bioluminescence_dict
