"""
The galleria_detction.py file for galleria_imaging

Contains the main functions for detecting galleria
"""

import matplotlib.pyplot as plt
from skimage.measure import regionprops
import numpy as np
from skimage.filters import sobel, threshold_yen, gaussian
import scipy.ndimage as ndi
from scipy.stats import ttest_ind

def detect_galleria(img, labelled_wells):
    """
    Runs the main galleria detection functions

    Parameters
    ------
    img : 2D array
        The brightfield image
    labelled_wells : ndi.labelled image
        An image where each unique value from 1 and above represents a detected well

    returns
    ------
    well_dict : dictionary
        dictionary where the key is the well number and the value is a 2d array the
        size of a well, with galleria labelled in side
    labelled_gall : labelled array
        Array where each label represents a detected galleria with the label the same as
        the well number. The rest of the array, including the wells, are 0.
    """

    well_dict = {}
    for region in regionprops(labelled_wells, intensity_image=img):
        well_dict[region.label] = detect_galleria_in_well(region.intensity_image)
    labelled_gall = map_galleria(labelled_wells, well_dict)

    return well_dict, labelled_gall

def map_galleria(labelled_wells, galleria_dict):
    """
    Creates an image with the galleria labelled

    Parameters
    ------
    labelled_wells : ndi.image
        ndi labelled image with each label representing a well location
    galleria_dict : dict
        Dictionary where the key is the well number and the value is a 2d array the
        size of a well, with galleria labelled in side

    returns
    ------
    labelled_gall : labelled array
        Array where each label represents a detected galleria with the label the same as
        the well number. The rest of the array, including the wells, are 0.
    """

    labelled_gall = np.zeros(labelled_wells.shape)

    for region in regionprops(labelled_wells):
        # get the well
        gall_in_well = galleria_dict[region.label]
        # change the label
        gall_in_well[gall_in_well == 1] = region.label
        # label the galleria in the correct location on the whole image
        labelled_gall[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] = gall_in_well

    return labelled_gall

def detect_galleria_in_cropped_well(well):
    sobel_img = sobel(well)
    blurred = gaussian(sobel_img, sigma=2.0)
    thresh = threshold_yen(well)
    dark_spots = np.array((well < thresh).nonzero()).T

    bool_mask = np.zeros(well.shape, dtype=np.bool)
    #bool_mask[tuple(light_spots.T)] = True
    bool_mask[tuple(dark_spots.T)] = True
    seed_mask, num_seeds = ndi.label(bool_mask)

    from skimage import morphology
    ws = morphology.watershed(blurred, seed_mask)

    fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(well)
    ax[1].imshow(well)
    ax[1].plot(dark_spots[:, 1], dark_spots[:, 0], 'o')
    ax[2].imshow(sobel_img)
    ax[3].imshow(blurred)
    ax[4].imshow(ws)
    plt.show()


def detect_galleria_in_well(well):
    """
    Function for detecting galleria within a well

    Parameters
    ------
    well : ndarray
        A 2d array of a well

    Returns
    ------
    result : labelled array
        Array the same size as the well where the galleria is labelled 1 and
        the background 0
    """
    # First lets see if any of the edges need cropping
    top, bottom, left, right = find_edges_to_crop(well)
    well_cropped = well[top:(well.shape[0]-bottom),left:(well.shape[1]-right)]
    # TODO: may want to remove this next median filter step and instead have a filtering
    # based on distance transform later on - this reduce the detected single pixel lines

    detect_galleria_in_cropped_well(well_cropped)

    well_cropped = ndi.median_filter(well_cropped, size=5)
    # Threshold the image
    thresh = threshold_yen(well_cropped)
    gall = well_cropped > thresh
    # Label the image
    gall, _ = ndi.label(gall)
    # we now need to put the labels back in to the original sized well
    labelled_gall = np.zeros(well.shape)
    labelled_gall[top:(labelled_gall.shape[0]-bottom),left:(labelled_gall.shape[1]-right)] = gall
    max_area_lab = None
    max_area = 0
    # Loop through labels
    for lab in list(np.unique(labelled_gall)):
        if lab == 0: # this is the background
            continue
        # find the label with the maximum area
        if np.sum( labelled_gall == lab ) > max_area:
            max_area = np.sum( labelled_gall == lab )
            max_area_lab = lab
    # Create a well with only the galleria labelled
    result = np.zeros(well.shape)
    result[labelled_gall == max_area_lab] = 1

    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(well_cropped)
    ax[1].imshow(labelled_gall)
    ax[2].imshow(result)
    plt.show()

    """
    return result


def find_edges_to_crop(well):
    """
    Determines if the well edges have been included and returns what needs cropping

    Parameters
    ------
    well : ndarray
        The 2D well profile

    Returns
    ------
    top : int
        The number of rows from the top that need cropping
    bottom : int
        The number of rows from the bottom that need cropping
    left : int
        The number of columns from the left that need cropping
    right : int
        The number of columns from the right that need cropping
    """
    # loop through top 3 rows
    for top in range(1, int(well.shape[0]/10)):
        # get row (need -1 as first row)
        rows_top = well[top-1].flatten()
        # compare the arrays and determine if row should be removed
        remove = compare_arrays(well.flatten(), rows_top)
        if remove is False:
            # if we don't remove it we set top to the previous loop and break the loop
            top -= 1
            break
    # repeat for left column, bottom rows and right column, respectively.
    for left in range(1, int(well.shape[1]/20)):
        cols_left = well[:, (left-1)].flatten()
        remove = compare_arrays(well.flatten(), cols_left)
        if remove is False:
            left -= 1
            break
    for bottom in range(1, int(well.shape[0]/10)):
        rows_bottom = well[-(bottom)].flatten()
        remove = compare_arrays(well.flatten(), rows_bottom)
        if remove is False:
            bottom -= 1
            break
    for right in range(1, int(well.shape[1]/20)):
        cols_right = well[:, -(right)].flatten()
        remove = compare_arrays(well.flatten(), cols_right)
        if remove is False:
            right -= 1
            break
    return top, bottom, left, right

def compare_arrays(array1, array2, p_val=0.001):
    """
    Runs a ttest and determines if the arrays are significantly different
    """
    # Run t test and assume non equal variance
    t_val, prob = ttest_ind(array1, array2, equal_var=False)
    # if the t_val is positive we know we want to keep the row/column as it means
    # the average (mean) is higher in the whole sample - so it isn't a well edge
    if t_val > 0:
        return False
    if prob > p_val:
        return False
    return True
