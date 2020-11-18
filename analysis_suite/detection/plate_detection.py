"""
The plate_detction.py file for galleria_imaging

Contains the main functions for detecting plates
"""

import matplotlib.pyplot as plt
import skimage.filters as skfilt
from scipy.signal import find_peaks
import skimage.measure as skmeas
import scipy.ndimage as ndi
from scipy import stats
from skimage.measure import regionprops
import numpy as np
from skimage.transform import probabilistic_hough_line
import skimage.morphology as skmorph
import math
from skimage.transform import resize
#from skimage.feature import canny

from analysis_suite.plate_dimensions import Plate

def straighten_plate(img, angle): # pragma: no cover
    """
    Takes the original image, determines the angle the plate is rotated and
    counter rotates it to straighen up
    Parameters
    ------
    img : ndarray
        2D array containing the original image
    angle : float
        The angle to rotate by
    Return
    ------
    rotated_img : ndarray
        2D array containing the rotated image
    """
    rotated_img = ndi.rotate(img, angle, mode='reflect')

    return rotated_img

def detect_plate_rotation(img):
    """
    Takes the original image, use a probabilistic hough line transform to
    identify straight lines around the plate. Determines the median angle in order to
    straighten them up

    Parameters
    ------
    img : ndarray
        2D array containing the original image

    Returns
    ------
    angle : float
        The angle at which the plate needs to be counter rotated
    """

    ## TODO: try loop - or try detecting whole plate

    # First we need may need to the image to ~400x400 pixels in order to speed up the probabilistic hough transform
    resized_img = resize_image(img)
    # perform sobel filter and threshold to detect ridges
    edges = skfilt.sobel(resized_img)
    ridges = edges > skfilt.threshold_otsu(edges)
    # perform probab hough transform to identify the lines
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    line_length = int(resized_img.shape[0] / 2)
    lines = probabilistic_hough_line(ridges, threshold=int(line_length/2), line_length=line_length,
                    line_gap=int(line_length/40), theta=tested_angles)

    # lets loop through the results
    angles = []
    for line in lines:
        # first get x and y points and determine the angle
        p0, p1 = line
        x_vals = [p0[0], p1[0]]
        y_vals = [p0[1], p1[1]]
        angle = math.degrees(math.atan2(y_vals[0] - y_vals[1], x_vals[0] - x_vals[1]))
        # some angles may be ~90, ~180 or ~270, so we need to adjust them so all of them
        # are ~0
        angle = adjust_angle(angle)
        if angle is not False:
            # Keep all the correct angles
            angles.append(angle)
    if len(angles) == 0:
        return 0 # # TODO: need to check why its 0 and improve above algorithm
    # rotate the image by the median angle
    angle = np.median(np.array(angles))

    return angle

def resize_image(img):
    """
    Adjusts the size of the image so either x or y is approximately 300-500 pixels

    Parameters
    ------
    img : ndarray
        2D array containing the original image

    Returns
    ------
    resized_img : ndarray
        Adjusted 2d Array
    factor : int
        The scale factor used to adjust the image
    """
    for i in range(1, 6):
        new_y = int(img.shape[0] / i)
        new_x = int(img.shape[1] / i)
        if ((300 <= new_y <= 500)) or ((300 <= new_x <= 500)):
            resized_img = resize(img, (new_y, new_x))
            return resized_img
    return img

def adjust_angle(angle):
    """
    Takes an angle and continuously removes 90 degrees until it falls with in a given range

    Parameters
    ------
    angle : int
        The angle to adjust

    Returns
    ------
    angle : int (or False)
        The adjusted angle, if the new angle can't be adjusted to within the specified range,
        returns False
    """
    # iterate through and take off 90 degrees each time
    for i in range(5):
        # not on the first loop
        if i > 0:
            angle -= 90
        # if we get it between -5 and -5 return the adjusted angle
        if (-5 < angle) and (angle < 5):
            return angle
        # if its less than -5 then its probably an incorrect line so return False
        elif angle < -5:
            return False

def detect_plate(img, plate_type=None):
    """
    Takes a np.array containing the image data and detects the main outline of the plate

    Parameters
    ------
    img : ndarray
        2D array containing the image data
    plate_type : str
        the type of plate being detected

    Returns
    ------
    labelled_wells : 2D array
        An array which has been labelled using ndi, with each well given a unique ID
    labelled_plate: 2D array
        An array which has been labelled using ndi, with the whole plate face (excluding wells) labelled as 1
    """

    # Generate plate
    currentplate = Plate(plate_type = plate_type)


    if "hex" in plate_type:
        # Because the Hexagonal plates are staggered and not straight ridges between wells, the
        # start/end of the plates are really clear - lets try and detect them directly
        start_x, end_x, start_y, end_y = get_corners_from_edges(img)
    else:
        start_x, start_y, x_gap, y_gap = get_first_well_and_gaps(img, currentplate._no_rows, currentplate._no_columns, plate_type=plate_type)
        if None in [start_x, start_y, x_gap, y_gap]:
            return None, None
        start_x, end_x, start_y, end_y = currentplate.get_plate_corners(start_x, start_y, x_gap, y_gap)


    # create mask the size of the bbox
    plate_mask = np.zeros(((end_y - start_y), (end_x - start_x)))
    # locate the wells
    currentplate.locate_wells(plate_mask, plate_type=plate_type)
    # If the first wells have been missed the plate is likely to still be the right size, but be slightly off
    # this can mean that the mask will be "stamped" off the edge of the image - as well as causing an error
    # this is wrong! lets check and move the mask back in the image (by the x/y gaps) if needed.

    start_x, end_x, start_y, end_y = move_plate_mask(start_x, end_x, x_gap, start_y, end_y, y_gap, img)

    # occasionally it still starts on the second row or the second column. The intensity of the plate mask when overlaid
    # on the image will be greatest when it is correctly aligned, so we can shift it iteratively to find the correct place if it is slightly out
    start_x, end_x, start_y, end_y = check_alignment_by_intensity(start_x, end_x, x_gap, start_y, end_y, y_gap, img)


    # create an image the same size as the original image and set the plate to the correct location
    # to match where the plate is found on the original image
    masked_wells = np.zeros(img.shape)
    masked_wells[start_y : end_y, start_x : end_x] = currentplate.plate

    masked_plate = np.copy(masked_wells)
    masked_plate[start_y : end_y, start_x : end_x] = 1

    labelled_wells = skmeas.label(masked_wells, return_num=False)
    labelled_plate = skmeas.label(masked_plate, return_num=False)

    return labelled_wells, labelled_plate

def check_alignment_by_intensity(start_x, end_x, gap_x, start_y, end_y, gap_y, img):
    """
    Takes the plate positions and moves it up and left iteratively, measuring the image intensity at each point
    when the mask has the best intensity, this is likely the correct position

    Parameters
    ------
    Parameters
    ------
    start_x : int
        The x coordinate of the left hand side of the plate
    end_x : int
        The x coordinate of the right hand side of the plate
    gap_x : int
        The gap in pixels between the start of each well (x axis)
    start_y : int
        The y coordinate of the top of the plate
    end_y : int
        The y coordinate of the bottom of the plate
    gap_y : int
        The gap in pixels between the start of each well (y axis)
    img : 2D array
        Original image

    Returns
    ------
    start_x : int
        The x coordinate of the left hand side of the plate
    end_x : int
        The x coordinate of the right hand side of the plate
    start_y : int
        The y coordinate of the top of the plate
    end_y : int
        The y coordinate of the bottom of the plate
    """
    # create a mask
    mask = np.zeros(img.shape)
    # set the plate to 1
    mask[start_y:end_y, start_x:end_x] = 1
    # initially set our first x and y values as our best and determine the mean intensity
    best_start_x = start_x
    best_end_x = end_x
    best_start_y = start_y
    best_end_y = end_y
    best_int_mean = np.mean(img[mask==1])

    # gradually shift the plate to the left and measure the intensity
    for n in range(1, 6):
        temp_start_x = start_x - int((n * gap_x))
        temp_end_x = end_x - int((n * gap_x))
        if temp_start_x > 0:
            mask = np.zeros(img.shape)
            mask[start_y:end_y, temp_start_x:temp_end_x] = 1
            intensity = np.sum(img[mask==1])
            int_mean = np.mean(img[mask==1])
            # if the intensity is better then replace the best values
            if int_mean > best_int_mean:
                best_int_mean = int_mean
                best_start_x = temp_start_x
                best_end_x = temp_end_x

    # repeat for y
    for n in range(1, 6):
        temp_start_y = start_y - int((n * gap_y))
        temp_end_y = end_y - int((n * gap_y))
        if temp_start_y > 0:
            mask = np.zeros(img.shape)
            mask[temp_start_y:temp_end_y, best_start_x:best_end_x] = 1
            intensity = np.sum(img[mask==1])
            int_mean = np.mean(img[mask==1])
            if int_mean > best_int_mean:
                best_int_mean = int_mean
                best_start_y = temp_start_y
                best_end_y = temp_end_y

    return best_start_x, best_end_x, best_start_y, best_end_y


def move_plate_mask(start_x, end_x, gap_x, start_y, end_y, gap_y, img):
    """
    Takes the plate positions and determines if the plate mask will fit in the image

    Parameters
    ------
    start_x : int
        The x coordinate of the left hand side of the plate
    end_x : int
        The x coordinate of the right hand side of the plate
    gap_x : int
        The gap in pixels between the start of each well (x axis)
    start_y : int
        The y coordinate of the top of the plate
    end_y : int
        The y coordinate of the bottom of the plate
    gap_y : int
        The gap in pixels between the start of each well (y axis)
    img : 2D array
        Original image

    Returns
    ------
    start_x : int
        The x coordinate of the left hand side of the plate
    end_x : int
        The x coordinate of the right hand side of the plate
    start_y : int
        The y coordinate of the top of the plate
    end_y : int
        The y coordinate of the bottom of the plate
    """

    poss_x_start, poss_x_end, poss_y_start, poss_y_end = get_corners_from_edges(img, max=False)
    if ((start_x - 3) <= poss_x_start <= (start_x + 3)) and ((start_y - 3) <= poss_y_start <= (start_y + 3)):
        return int(start_x), int(end_x), int(start_y), int(end_y)

    img_shape = img.shape
    ## First lets make sure all masks will be inside the image area by moving them by the
    # known gaps if we need to
    for n in range(0, 5):
        if (end_x < img_shape[1]):
            continue
        end_x -= gap_x
        start_x -= gap_x
    for n in range(0, 5):
        if (end_y < img_shape[0]):
            continue
        end_y -= gap_y
        start_y -= gap_y

    # Now loop through and see if any shifts mean we hit the possible starts we identified
    # N.B. We probably only need to do this for y axis as the column splits are a lot more "obvious"
    # so we usually detect it quite accurately
    """
    for n in range(0, 5):
        if ((start_x - (gap_x*n) - 5) <= poss_x_start <= (start_x - (gap_x*n) + 5)):
            start_x -= (gap_x*n)
            end_x -= (gap_x*n)
            continue
    """
    for n in range(0, 5):
        if ((start_y - (gap_y*n) - 5) <= poss_y_start <= (start_y - (gap_y*n) + 5)):
            start_y -= (gap_y*n)
            end_y -= (gap_y*n)
            continue

    return int(start_x), int(end_x), int(start_y), int(end_y)


def get_corners_from_edges(img, perc=90, max=True):
    """
    Function for detecting the corners of a hexagonal plate. Because the
    plate isn't consistent in well spacing/layout like a rectangular plate the
    edges are more "obvious" when looking at peaks of the 1D profile.

    Parameters
    ------
    img : ndarray
        2D array containing the image data
    perc : int, optional

    Returns
    ------
    start_x : int
        The x coordinate of the left hand side of the plate
    end_x : int
        The x coordinate of the righ hand side of the plate
    start_y : int
        The y coordinate of the top of the plate
    end_y : int
        The y coordinate of the bottom of the plate
    """
    # get the mean of the x axis (1D profile)
    horiz_mean = np.mean(img, axis=0)
    # get the difference between adjacent pixels to exaggerate changes
    profile_diff = np.diff(horiz_mean)
    # get the peaks and troughs and their respective prominence
    troughs, trough_prom = get_peaks_and_prominence(np.negative(profile_diff), perc=perc)
    peaks, prominence = get_peaks_and_prominence(profile_diff, perc=perc)

    if max:
        # the max trough is the start of the plate and the max peak is the end of the plate
        start_x = troughs[np.where(trough_prom == np.max(trough_prom))[0][0]]
        end_x = peaks[np.where(prominence == np.max(prominence))[0][0]]
    else:
        troughs, trough_prom = reject_outliers(troughs, trough_prom)
        start_x = troughs[0]
        peaks, prominence = reject_outliers(peaks, prominence)
        end_x = peaks[-1]


    # repeat for y_axis
    vert_mean = np.mean(img, axis=1)
    profile_diff = np.diff(vert_mean)
    troughs, trough_prom = get_peaks_and_prominence(np.negative(profile_diff), perc=perc)
    peaks, prominence = get_peaks_and_prominence(profile_diff, perc=perc)

    if max:
        start_y = troughs[np.where(trough_prom == np.max(trough_prom))[0][0]]
        end_y = peaks[np.where(prominence == np.max(prominence))[0][0]]
    else:
        troughs, trough_prom = reject_outliers(troughs, trough_prom)
        start_y = troughs[0]
        peaks, prominence = reject_outliers(peaks, prominence)
        end_y = peaks[-1]

    return start_x, end_x, start_y, end_y


def get_first_well_and_gaps(img, nrows, ncols, plate_type=None):
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
    start_x, x_gap = find_first_well(profile_diff, ncols+1, perc=90)
    vert_mean = np.mean(img, axis=1)
    profile_diff = np.diff(vert_mean)
    start_y, y_gap = find_first_well(profile_diff, nrows+1, perc=90)

    return start_x, start_y, x_gap, y_gap

def get_peaks_and_prominence(profile, perc=90):
    """
    Takes the 1D profile and returns peaks and their respective prominence

    Parameters
    ------
    profile : ndarray
        1 dimensional array
    perc : int (optional)
        The percentile in which the peaks are considered prominent

    Returns
    ------
    peaks : array
        1D array with the coords of each peak
    prominence : array
        1D array with the respective prominence of each peak
    """
    # we want to find when the image goes from "light" to "dark" so need to flip the values
    profile = np.negative(profile)
    # find the peaks
    peaks, peak_dict = find_peaks(profile, distance=20, prominence=np.percentile(np.absolute(profile), perc))
    peaks = np.array(peaks)
    #Lets filter based on prominence to only get the number of peaks we need (number of rows or columns, respectively)
    prominence = np.array(peak_dict["prominences"])
    """
    import matplotlib.pyplot as plt
    x = profile
    peaks1, _ = find_peaks(x, distance=20,prominence=np.percentile(np.absolute(profile), perc))
    peaks2, _ = find_peaks(x, prominence=1)
    peaks3, _ = find_peaks(x, width=10)
    peaks4, _ = find_peaks(x, threshold=0.4)     # Required vertical distance to its direct neighbouring samples, pretty useless
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(peaks1, x[peaks1], "xr"); plt.plot(x); plt.legend(['distance'])
    plt.subplot(2, 2, 2)
    plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['prominence'])
    plt.subplot(2, 2, 3)
    plt.plot(peaks3, x[peaks3], "vg"); plt.plot(x); plt.legend(['width'])
    plt.subplot(2, 2, 4)
    plt.plot(peaks4, x[peaks4], "xk"); plt.plot(x); plt.legend(['threshold'])
    plt.show()
    print(peaks, prominence)
    """
    return peaks, prominence

def find_first_well(profile, num_peaks, perc=90):
    """
    Takes the 1D profile, determines the difference between points and identifies the major
    peaks, then identifies the first one which is regularly spaced

    Parameters
    ------
    profile : ndarray
        1 dimensional array
    num_peaks : int
        The number of peaks we are looking for
    perc : int (optional)
        The percentile in which the peaks are considered prominent

    Returns
    ------
    coord : int
        The coordinate for the first well (singular as only 1 dimensional)
    med : int
        The spacing of the wells
    """
    peaks, prominence = get_peaks_and_prominence(profile, perc=perc)

    peaks, prominence = reject_outliers(peaks, prominence)

    # Now we have filtered for outliers (which should remove major peaks like the start/end of the plate)
    # we remove all peaks with prominence less than 20% of the max prominence
    # calculate half the max and create a mask which can be used to filter
    half_max_prom = max(prominence) * 0.20
    mask = prominence > half_max_prom

    # We dont want to remove too many peaks, we need a minimum of 5 to carry on
    if sum(mask) > 5:
        # use mask to keep only those above half max
        peaks = peaks[mask]
        prominence = prominence[mask]

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
    consistent_gaps = np.array(((med-(med*0.1)) < diff) & (diff < (med+(med*0.1))))
    if sum(consistent_gaps) == 0:
        return None, None
    # get the first gap
    first_gap = np.where(consistent_gaps == True)[0][0]
    coord = best_peaks[first_gap]
    # lets update the median using only the consistent_gaps
    med = np.median(diff[np.where(consistent_gaps == True)])
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
                coord = (peak + poss) / 2
                continue

    return coord, med

def reject_outliers(peaks, prominence,  m=2):
    """
    A simple function that removes outliers based on peak prominence

    Parameters
    ------
    peaks : numpy array
        A 1D array with the corresponding peak locations
    prominence : numpy array
        A 1D array containing the prominence of peaks
    m : int, optional
        The number of standard deviations used for identifying outliers (default : 2)

    Returns
    data : numpy array
        A 1D array containing the filtered prominence of peaks
    peaks : numpy array
        A 1D array with the filtered corresponding peak locations
    """
    peaks = peaks[abs(prominence - np.mean(prominence)) < m * np.std(prominence)]
    prominence = prominence[abs(prominence - np.mean(prominence)) < m * np.std(prominence)]
    return peaks, prominence
