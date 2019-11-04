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

    labelled_plate : binary array
        Binary array, where wells and the background are 0 and the main plate is 1
    fluo_image : array
        Array of the image containing the fluorescence information
    """

    # need to invert the 1s and 0s
    labelled_plate[labelled_plate == 1] = 2
    labelled_plate[labelled_plate == 0] = 1
    labelled_plate[labelled_plate == 2] = 0

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
    """
    # TODO: need to improve this function!!!

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

    """
    plt.figure(figsize=(16, 12))
    plt.imshow(img, cmap='gray')

    # Create a label  and plot the contours

    plt.contour(lbl, levels=[0.5], colors=["r"], linewidths=[4, ])
    plt.show()
    """
    return lbl

    """
    plt.figure()
    plt.subplot(151)
    plt.imshow(img, cmap="gray")
    plt.subplot(152)
    plt.imshow(filt)
    plt.subplot(153)
    plt.imshow(ridges)
    plt.subplot(154)
    plt.imshow(lbl_filled)
    plt.subplot(155)
    plt.imshow(masked_image)
    plt.show()
    """
