"""
Contains functions that generate fake plate images for testing
"""

import matplotlib.pyplot as plt
import numpy as np
import analysis_suite.tests.galleria_creator as galleria
import skimage.measure as skmeas
from analysis_suite.plate_dimensions import Plate


def generate_fake_plate_image(max_bkg = 7000, max_pixel = 25000, plate_length = 280, plate_ratio= 0.67, n_ver=8, hex=False):
    """
    Generates a fake image with a plate that can be used to test detection

    Parameters
    ------
    max_bkg : int
        Maximum pixel intensity for the background
    max_pixel : int
        Average maximum pixel used for gaussian randomly generate pixel values
    plate_length : int
        Length of the plate (y-axis)
    plate_ratio : float
        Ratio of length / width

    Returns
    ------
    image : numpy array
        A numpy array representing an image of a plate with galleria in wells
    """
    plate_shape = tuple((plate_length, round(plate_length / plate_ratio)))

    # First generate the background of random pixel intensities
    img_shape = ((346, 464))#tuple(img_shape)
    image = np.random.randint(max_bkg, size=img_shape)

    """
    # Generate the base of the plate - this needs to be slightly bigger (i.e. 5 pixels) than
    # the main plate face and lower pixel intensity
    base_shape = tuple(np.array(plate_shape) + 10)
    plate_base = generate_fake_plate(plate_shape=base_shape, max_pixel=(max_pixel*0.75), base=True)

    # Need to find the difference between the plate base and the image outline to locate it centrally
    x_half = int((img_shape[1] - base_shape[1]) / 2)
    y_half = int((img_shape[0] - base_shape[0]) / 2)

    # add the base shell to the background array
    image[y_half:y_half+plate_base.shape[0],x_half:x_half+plate_base.shape[1]] = plate_base
    """

    # Generate the plate itself - the corner cut offs need to be the same as the base
    plate_shell = generate_fake_plate(plate_shape=plate_shape, max_pixel=max_pixel, well_pixel=max_bkg,
                    corners=False, n_ver=n_ver, hex=hex)

    # Need to find the difference between the plate and the image shapes so we can
    # put the plate in the centre of the image
    x_half = int((img_shape[1] - plate_shape[1]) / 2)
    y_half = int((img_shape[0] - plate_shape[0]) / 2)

    # add the plate shell to the background array
    image[y_half:y_half+plate_shell.shape[0],x_half:x_half+plate_shell.shape[1]]  = plate_shell

    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()
    """
    return image

def generate_fake_plate(corner_pixel = 3500, well_pixel = 7000, max_pixel = 25000, plate_shape=(300, 400), base=False, corners=False, n_ver=8, hex=False):
    """
    Generates a fake plate with given dimensions and corner cut offs in two corners

    Parameters
    ------
    corner_pixel : int
        Value to which the corner cut offs need to be normalised around
    well_pixel : int
        Value to which the wells need to be normalised around
    max_pixel : int
        Average maximum pixel used for gaussian randomly generate pixel values
    plate_shape : tuple
        Rough external dimensions of plate to be created
    base : boolean
        If True then it returns the whole shape before corner cutting/wells added
    """
    # Generate normal distribution around 1 and then multiply this by the max pixel
    # to get a random distribution around the right pixel intensity
    plate_shell = (np.random.normal(loc=1, scale=0.05 , size=plate_shape)*max_pixel)
    if base:
        return plate_shell
    if corners:
        # Cut corners off the top left and bottom left of the plate
        plate_shell = cut_plate_corner(plate_shell, corner_pixel=corner_pixel, corner="top_left")
        plate_shell = cut_plate_corner(plate_shell, corner_pixel=corner_pixel, corner="bottom_left")

    # Add evenly distributed wells
    if hex:
        plate_shell = add_hexagonal_wells_in_plate(plate_shell, well_pixel=well_pixel, galleria_pixel=max_pixel, n_ver=n_ver)
    else:
        plate_shell = add_rectangular_wells_in_plate(plate_shell, well_pixel=well_pixel, galleria_pixel=max_pixel, n_ver=n_ver)

    import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(plate_shell)
    #plt.show()

    return plate_shell

def add_hexagonal_wells_in_plate(plate_shell, well_pixel=250, galleria_pixel=750, n_hor = 5, n_ver = 8):
    """
    Adds hexagonal wells in to the fake plate

    Parameters
    ------
    plate_shell : ndarry
        2D numpy array in which the wells need to be added
    well_pixel : int
        Maximum pixel intensity for the wells
    n_hor : int
        number of horizontal wells (essentially number of columns)
    n_ver : int
        number of vertical wells (essentially number of rows)

    Returns
    ------
    plate_shell : ndarray
        The plate with the wells added
    """

    plate_shape = plate_shell.shape
    # lets give a gap of 10 % from a vertical edge for our first column
    # then calculate the spacing between the start of each column
    hor_end_gap = plate_shape[1] / 10
    hor_spacing = int((plate_shape[1] - hor_end_gap) / 11)

    # repeat for rows
    ver_end_gap = plate_shape[0] / 40
    ver_spacing = int((plate_shape[0] - ver_end_gap) / 5)

    # Generate plate
    currentplate = Plate(plate_type = "hex50")

    start_x, end_x, start_y, end_y = currentplate.get_plate_corners(hor_end_gap, ver_end_gap, hor_spacing, ver_spacing)

    # create mask the size of the bbox
    plate_mask = np.zeros(((end_y - start_y), (end_x - start_x)))

    # locate the wells
    currentplate.locate_wells(plate_mask)

    plate = currentplate.plate

    labelled_wells = skmeas.label(plate, return_num=False)

    for region in skmeas.regionprops(labelled_wells):
        if region.label==0:
            continue
        well_height = region.bbox[2] - region.bbox[0]
        well_width = region.bbox[3] - region.bbox[1]
        # generate a well of random values
        well_ = np.random.normal(loc=1, scale=0.05, size=(well_height, well_width)) * well_pixel
        # set this to the correct area of the plate
        well_ = galleria.well_with_galleria(well_, galleria_pixel = galleria_pixel)
        plate[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] = well_

    plate[labelled_wells == 0] = 0
    plate_shell = (np.random.normal(loc=1, scale=0.05 , size=plate.shape)*galleria_pixel)
    for region in skmeas.regionprops(labelled_wells):
        if region.label==0:
            continue
        plate_shell[labelled_wells == region.label] = 0
    plate_shell += plate

    return plate_shell



def add_rectangular_wells_in_plate(plate_shell, well_pixel=250, galleria_pixel=750, n_hor = 5, n_ver = 8):
    """
    Adds rectangular wells in to the fake plate

    Parameters
    ------
    plate_shell : ndarry
        2D numpy array in which the wells need to be added
    well_pixel : int
        Maximum pixel intensity for the wells
    n_hor : int
        number of horizontal wells (essentially number of columns)
    n_ver : int
        number of vertical wells (essentially number of rows)

    Returns
    ------
    plate_shell : ndarray
        The plate with the wells added
    """

    plate_shape = plate_shell.shape
    # lets give a gap of 10 % from a vertical edge for our first column
    # then calculate the spacing between the start of each column
    hor_end_gap = plate_shape[1] / 10
    hor_spacing = int((plate_shape[1] - hor_end_gap) / n_hor)

    # repeat for rows
    ver_end_gap = plate_shape[0] / 40
    ver_spacing = int((plate_shape[0] - ver_end_gap) / n_ver)

    # calculate the column width
    well_width = int(hor_spacing - (hor_spacing / 5))
    # and well height
    well_height = int(ver_spacing - (ver_spacing / 5))

    # Need to loop until we have the x start position of each column
    x_row_starts = []
    current_x = hor_end_gap
    while current_x <= (plate_shape[1] - hor_spacing):
        x_row_starts.append(int(current_x))
        current_x = current_x + hor_spacing

    # repeat for the y start position for each row
    y_row_starts = []
    current_y = ver_end_gap
    while current_y <= (plate_shape[0] - ver_spacing):
        y_row_starts.append(int(current_y))
        current_y = current_y + ver_spacing

    for x_val in x_row_starts:
        for y_val in y_row_starts:
            # generate a well of random values
            well_ = np.random.normal(loc=1, scale=0.05, size=(well_height, well_width)) * well_pixel
            # set this to the correct area of the plate
            well_ = galleria.well_with_galleria(well_, galleria_pixel = galleria_pixel)
            plate_shell[y_val:y_val+well_height, x_val:x_val+well_width] = well_

    return plate_shell

def cut_plate_corner(plate_shell, corner_pixel = 250, num_pixels=15, corner="top_left"):
    """
    "Cuts" the corner of a numpy array by setting triangles in the corner to
    random integers up to the maximum specified

    Parameters
    ------

    plate_shell : ndarry
        2D numpy array in which the corners need to be "cut"
    corner_pixel : int
        Value to which the corner cut offs need to be normalised around
    num_pixels : int
        Number of pixels to come in from the corners to create the cut off
    corner : str
        Specifies which corner to "cut" - "top_left", "top_right", "bottom_left", "bottom_right"

    Returns
    ------
    plate_shell : ndarry
        2D numpy array in which the corners have been "cut"
    """

    array_shape = plate_shell.shape
    # First we need to get the correct indices depending on which corner
    if "left" in corner:
        x_vals = list(range(0, num_pixels))
    else:
        x_vals = list(range(array_shape[1]-num_pixels, array_shape[1]))
        x_vals = sorted(x_vals, reverse=True)

    if "top" in corner:
        y_vals = list(range(0, num_pixels))
    else:
        y_vals = list(range(array_shape[0]-num_pixels, array_shape[0]))
        y_vals = sorted(y_vals, reverse=True)

    # We now loop through the x coordinates but go one less along the y coordinates
    # each time. For example 3 pixels would give: (0,0) (0, 1) (0, 2) (1, 0) (1, 1) (2, 0)
    coords = []
    for n, x_val in enumerate(x_vals, start=0):
        for y_val in y_vals[:(len(y_vals) - n)]:
            coords.append((y_val, x_val))

    # Set each coordinate in the list to a random integer in the same range as the background
    for coo_ in coords:
        plate_shell[coo_] = np.random.normal(loc=1, scale=0.05) * corner_pixel

    return plate_shell
