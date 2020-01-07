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

    start_x, start_y, x_gap, y_gap = get_first_well_and_gaps(img, currentplate._no_rows, currentplate._no_columns)

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

def get_first_well_and_gaps(img, nrows, ncols):
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
    plt.figure()
    plt.imshow(img)
    # get the mean of the x axis (1D profile)
    horiz_mean = np.mean(img, axis=0)
    # get the difference between adjacent pixels to exaggerate changes
    profile_diff = np.diff(horiz_mean)
    # get the x location of the top left corner of the first well and the well spacing
    start_x, x_gap = find_first_well(profile_diff, ncols, perc=90)
    # repeat for y axis (rows of wells)
    vert_mean = np.mean(img, axis=1)
    profile_diff = np.diff(vert_mean)
    start_y, y_gap = find_first_well(profile_diff, nrows, perc=90)
    return start_x, start_y, x_gap, y_gap

def find_first_well(profile, num_peaks, perc=95):
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
    peaks, peak_dict = find_peaks(profile, distance=20, prominence=np.percentile(np.absolute(profile), perc))
    peaks = np.array(peaks)
    #Lets filter based on prominence to only get the number of peaks we need (number of rows or columns, respectively)
    prominence = np.array(peak_dict["prominences"])

    #remove all peaks with prominence less than 20% of the max prominence
    # calculate half the max and create a mask which can be used to filter
    half_max_prom = max(prominence) * 0.20
    mask = prominence > half_max_prom
    # We dont want to remove too many peaks, we need a minimum of 5 to carry on
    if sum(mask) > 5:
        # use mask to keep only those above half max
        peaks = peaks[mask]
        prominence = prominence[mask]

    ### # TODO: For the hexagonal we will get more peaks that the number of columns because of the stagger?!
    if len(prominence) > num_peaks:
        top_peaks = sorted(prominence, reverse=True)[:num_peaks]
    else:
        top_peaks = prominence
    ind_top_peaks = [ind for ind, prom in enumerate(prominence) if prom in top_peaks]
    best_peaks = [peaks[ind] for ind in ind_top_peaks]
    # find spacing between peaks and calculate the median change
    diff = np.diff(best_peaks)
    med = np.median(np.diff(best_peaks))

    # identify peaks which fall in a consistent gap range (added 20% to median to cover small variations)
    consistent_gaps = np.array(((med-(med*0.2)) < diff) & (diff < (med+(med*0.2))))
    # get the first gap
    first_gap = np.where(consistent_gaps == True)[0][0]
    coord = best_peaks[first_gap]

    # using the first identified peak and the calculated gap, lets interpolate where we may find
    # a missed peak
    possible_missed = [(coord - (med * n)) for n in range(1, 10) if (coord - (med * n) > 0)]

    # Now loop through the possible locations and see if any peaks were identified near there
    for poss in possible_missed:
        for peak in peaks:
            if peak == coord:
                continue
            # if within 3 pixels we replace it as the first peak
            if (poss - 3) <= peak <= (poss + 3):
                coord = peak
                continue

    return coord, med
