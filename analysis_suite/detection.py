"""
The detction.py file for galleria_imaging

Contains the main detection functions
"""

import matplotlib.pyplot as plt
import skimage.filters as skfilt
from scipy.signal import find_peaks
import skimage.measure as skmeas
import scipy.ndimage as ndi
from scipy import stats
from skimage.measure import regionprops
import numpy as np

from analysis_suite.plate_dimensions import Plate

def extract_biolum_values(labelled_wells, fluo_image):
    """
    Extracts bioluminescence values from the fluo image by over laying the detected plate

    Parameters
    ------
    labelled_wells : binary array
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

    fluo_image = np.array(fluo_image)

    bioluminescence_dict = {}

    for region in regionprops(labelled_wells):
        # get mean fluo from the well area in matching fluo image
        fluo = np.mean(fluo_image[tuple(np.array(region.coords).T)])
        bioluminescence_dict[region.label] = [region.area, fluo, fluo*region.area]

    return bioluminescence_dict

def detect_plate(img, plate_type=None):
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

    img = np.array(img)

    # Generate plate
    currentplate = Plate(plate_type = plate_type)

    start_x, start_y, x_gap, y_gap = get_first_well_and_gaps(img)

    start_x, end_x, start_y, end_y = currentplate.get_plate_corners(start_x, start_y, x_gap, y_gap)

    # create mask the size of the bbox
    plate_mask = np.zeros(((end_y - start_y), (end_x - start_x)))

    # locate the wells
    currentplate.locate_wells(plate_mask)

    # create an image the same size as the original image and set the plate to the correct location
    # to match where the plate is found on the original image
    masked_wells = np.zeros(img.shape)
    masked_wells[start_y : end_y, start_x : end_x] = currentplate.plate

    masked_plate = np.copy(masked_wells)
    masked_plate[start_y : end_y, start_x : end_x] = 1

    labelled_wells = skmeas.label(masked_wells, return_num=False)
    labelled_plate = skmeas.label(masked_plate, return_num=False)
    """
    plt.figure()
    plt.imshow(img)
    # loop through the labelled wels and plot the contours of each label
    for well_lab in range(1, labelled_wells.max() + 1):
        # create colour for contour
        col = plt.cm.gist_rainbow((well_lab / 9.1) % 1)
        # plot contour
        plt.contour(labelled_wells == well_lab, levels=[0.5], colors=[col])
        # get the location of the well and find the top right corner in order to add the text label
        bw0 = labelled_wells == well_lab
        pos0 = bw0.nonzero()
        pos = (np.min(pos0[0]), np.max(pos0[1]))
        plt.text(pos[1], pos[0], str(well_lab), color=col)
    # set the colour to one above the maximum number of the wells and use
    # it to plot the contour of the plate outline
    col = plt.cm.gist_rainbow(((labelled_wells.max()+1) / 9.1) % 1)
    plt.contour(labelled_plate == 1, levels=[0.5], colors=[col])
    plt.show()
    """
    return labelled_wells, labelled_plate

def get_first_well_and_gaps(img):
    """
    Converts the image into mean 1d profiles (vertical and horizontal, respectively).
    Then determines the difference between adjactent points, uses the upper 95 % to
    identify peaks based on prominence (scipy.signal.find_peaks) and takes the first and
    last peak as the start and end of the plate.

    Parameters
    ------
    img : Array
        An array containing the greyscale pixel values of the img

    Returns
    ------
    start_x : int
        The x coordinate for the top left and bottom left corners
    end_x : int
        The x coordinate for the top right and bottom right corners
    x_gao : int
        The spacing between horizontal wells (in pixels)
    y_gap : int
        The spacing between vertical wells (in pixels)
    """
    # get the mean of the x axis (1D profile)
    horiz_mean = np.mean(img, axis=0)
    # get the difference between adjacent pixels to exaggerate changes
    profile_diff = np.diff(horiz_mean)
    # get the x location of the top left corner of the first well and the well spacing
    start_x, x_gap = find_first_well_and_gap(profile_diff, perc=90)

    # repeat for y axis (rows of wells)
    vert_mean = np.mean(img, axis=1)
    profile_diff = np.diff(vert_mean)
    start_y, y_gap = find_first_well_and_gap(profile_diff, perc=90)
    return start_x, start_y, x_gap, y_gap

def find_first_well_and_gap(profile, perc=90):
    """
    Takes the 1D profile, determines the difference between points and identifies the major
    peaks, then identifies the first one which is regularly spaced

    Parameters
    ------
    profile : ndarray
        1 dimensional array
    perc : int (optional)
        The percentile in which the peaks are considered prominent

    Returns
    ------
    coord : int
        The coordinate for the first well (singular as only 1 dimensional)
    med : int
        The spacing of the wells
    """
    # we want to find when the image goes from "light" to "dark" so need to flip the values
    profile = np.negative(profile)
    # find the peaks
    peaks, _ = find_peaks(profile, distance=20, prominence=np.percentile(np.absolute(profile), perc))
    # find spacing between peaks and calculate the median change
    diff = np.diff(peaks)
    med = np.median(np.diff(peaks))
    # identify peaks which fall in a consistent gap range (added 20% to median to cover small variations)
    consistent_gaps = np.array(((med-(med*0.2)) < diff) & (diff < (med+(med*0.2))))
    # get the first gap
    first_gap = np.where(consistent_gaps == True)[0][0]
    coord = peaks[first_gap]

    return coord, med






######## Old unused functions, but want to push before deleting in case I want them in future

#def find_first_well_and_gap(profile):
    """
    Takes the 1D profile, determines the difference between points and identifies the first
    and last major peaks

    Parameters
    ------
    profile : np.array
        1D array representing the mean pixel value along a given axis

    Returns
    ------
    start : int
        The first prominent peak along the respective axis
    end : int
        The last prominent trough along the respective axis
    """

    # get the difference between adjactent points
#    profile_diff = np.diff(profile)

#    first, med = test_function_first_well(profile_diff)

    #print("IMPORTANT: %s, %s"%(first, med), flush=True)
    #return first, med

    #check_peaks(np.negative(profile_diff))
    #return start, end


def check_peaks(x, perc=90, end=False):

    if end:
        x = np.flipud(np.negative(x))

    peaks, _ = find_peaks(x, distance=20, prominence=np.percentile(np.absolute(x), perc))
    diff = np.diff(peaks)
    med = np.median(np.diff(peaks))
    mad = stats.median_absolute_deviation(np.diff(peaks))

    #if mad == 0:
#        mad = 1 #if it is too consistent (need to add a small mad just in case one is one pixel out)

    consistent_gaps = np.array(((med-(med*0.1)) < diff) & (diff < (med+(med*0.1))))
    first_gap = np.where(consistent_gaps == True)[0][0]
    #last_gap = np.where(consistent_gaps == True)[0][-1]

    peaks2, _ = find_peaks(x, prominence=np.percentile(np.absolute(x), perc))      # BEST!
    peaks3, _ = find_peaks(x, width=20,prominence=np.percentile(np.absolute(x), perc))
    peaks4, _ = find_peaks(x, threshold=0.4, prominence=np.percentile(np.absolute(x), perc))     # Required vertical distance to its direct neighbouring samples, pretty useless
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['distance'])
    #plt.subplot(2, 2, 2)
    #plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['prominence'])
    #plt.subplot(2, 2, 3)
    #plt.plot(peaks3, x[peaks3], "vg"); plt.plot(x); plt.legend(['width'])
    #plt.subplot(2, 2, 4)
    #plt.plot(peaks4, x[peaks4], "xk"); plt.plot(x); plt.legend(['threshold'])

    if first_gap == 0:
        plate_start = peaks[0]
    else:
        plate_start = peaks[first_gap-1]
    if end:
        return len(x) - plate_start
    else:
        return plate_start


"""
def get_best_peaks_by_ratio(start_x, end_x, start_y, end_y, xy_ratio, img_shape):
    x_dict = {}
    for x_st in start_x:
        for x_en in end_x:
            if x_en < x_st:
                continue
            if (x_en - x_st) < (img_shape[1] / 2):
                continue
            x_dict[(x_st, x_en)] = x_en - x_st
    y_dict = {}
    for y_st in start_y:
        for y_en in end_y:
            if y_en < y_st:
                continue
            y_dict[(y_st, y_en)] = y_en - y_st

    combined_dict = {}
    for key_x, value_x in x_dict.items():
        for key_y, value_y in y_dict.items():
            ratio = value_x / value_y
            combined_dict[(key_x, key_y)] = abs(xy_ratio - ratio)
    current_max = 0
    current_key = None
    for key, val in combined_dict.items():
        if val < 0.1:
            area = (key[0][1]-key[0][0]) * (key[1][1] - key[1][0])
            if area > current_max:
                current_max = area
                current_key = key

    best_combo = min(combined_dict, key=combined_dict.get)
    best_combo = current_key

    return best_combo[0][0], best_combo[0][1], best_combo[1][0], best_combo[1][1],
"""
