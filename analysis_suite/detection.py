"""
The detction.py file for galleria_imaging

Contains the main detection functions
"""

import matplotlib.pyplot as plt
import skimage.filters as skfilt
import skimage.measure as skmeas
import scipy.ndimage as ndi
from skimage.measure import regionprops
import numpy as np

from analysis_suite.plate_dimensions import Plate

def extract_biolum_values(labelled_plate, fluo_image):
    """
    Extracts bioluminescence values from the fluo image by over laying the detected plate

    Parameters
    ------
    labelled_plate : binary array
        Binary array, where wells and the background are 0 and the main plate is 1
    fluo_image : array
        Array of the image containing the fluorescence information

    Returns
    ------
    labelled_plate : labelled image
        ndi labelled image from skimage.ndimage - where each well has a unique number starting from 2
        as the background is also labelled
    bioluminescence_dict : dictionary
        Where key is the region label and value is a list of measurements [area, fluo, integarted fluo]
    """

    # need to invert the 1s and 0s
    labelled_plate[labelled_plate == 1] = 2
    labelled_plate[labelled_plate == 0] = 1
    labelled_plate[labelled_plate == 2] = 0

    # Label the plate
    labelled_plate = ndi.label(labelled_plate)[0]

    fluo_image = np.array(fluo_image)

    bioluminescence_dict = {}
    for region in regionprops(labelled_plate):
        # Remove background. # TODO:  This isn't good that its hard coded so
        #need to change and improve - currently temporary
        if region.area > 20000:
            labelled_plate[labelled_plate == region.label] = 0
            continue

    labelled_plate = ndi.label(labelled_plate)[0]
    for region in regionprops(labelled_plate):
        # get mean fluo from the well area in matching fluo image
        fluo = np.mean(fluo_image[tuple(np.array(region.coords).T)])
        bioluminescence_dict[region.label] = [region.area, fluo, fluo*region.area]

    return labelled_plate, bioluminescence_dict

def detect_plate(img):
    """
    Takes a np.array containing the image data and detects the main outline of the plate

    Parameters
    ------
    img : ndarray
        2D array containing the image data

    lbl : labelled image
        ndi labelled image from skimage.ndimage - where the main plate object is labelled
    """
    ### TODO: need to improve this function!!!

    # Perform sobel filter to detect ridges
    filt = skfilt.sobel(img)

    ridges = filt > skfilt.threshold_li(filt)

    # May want to filter away small areas?

    # Fill the whole plate
    lbl_filled = ndi.morphology.binary_fill_holes(ridges)

    ### # TODO: temporarily erode to make correct size
    lbl_filled = ndi.binary_erosion(lbl_filled, iterations=4)

    # convert boolean to binary
    lbl_filled = lbl_filled * 1

    # get regionprops of the detected plate
    region = regionprops(lbl_filled)[0]

    # create mask the size of the bbox - needs to be done better!!!
    plate_mask = np.zeros(((region.bbox[2] - region.bbox[0]), (region.bbox[3] - region.bbox[1])))

    # Generate plate
    currentplate = Plate(well_type = "rect")
    currentplate.locate_wells(plate_mask)

    # create an image the same size as the original image and set the plate to the correct location
    # to match where the plate is found on the original image
    masked_image = np.zeros(filt.shape)
    masked_image[region.bbox[0] : region.bbox[2], region.bbox[1] : region.bbox[3]] = currentplate.plate

    lbl = skmeas.label(masked_image, return_num=True)[0]

    return lbl
