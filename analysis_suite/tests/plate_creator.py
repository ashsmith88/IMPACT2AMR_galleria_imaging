"""
Contains functions that generate fake plate images for testing
"""

import matplotlib.pyplot as plt
import numpy as np
import analysis_suite.tests.galleria_creator as galleria

def generate_fake_plate_image(max_bkg = 250, max_pixel = 8000, img_shape=(346, 464), plate_shape=(320, 430)):
    """
    Generates a fake image with a plate that can be used to test detection

    Parameters
    ------
    max_bkg : int
        Maximum pixel intensity for the background
    max_pixel : int
        Average maximum pixel used for gaussian randomly generate pixel values
    img_shape : tuple
        Tuple containing the dimensions of the image to be created
    plate_shape : tuple
        Rough external dimensions of plate to be created
    """

    # First generate the background of random pixel intensities
    bkg = np.random.randint(max_bkg, size=img_shape)

    # Generate the base of the plate - this needs to be slightly bigger (i.e. 5 pixels) than
    # the main plate face and lower pixel intensity
    base_shape = tuple(np.array(plate_shape) + 10)
    plate_base = generate_fake_plate(plate_shape=base_shape, max_pixel=(max_pixel*0.75), base=True)

    # Need to find the difference between the plate base and the image outline to locate it centrally
    x_half = int((img_shape[1] - base_shape[1]) / 2)
    y_half = int((img_shape[0] - base_shape[0]) / 2)

    # add the base shell to the background array
    bkg[y_half:bkg.shape[0]-y_half,x_half:bkg.shape[1]-x_half] = plate_base

    # Generate the plate itself - the corner cut offs need to be the same as the base
    plate_shell = generate_fake_plate(plate_shape=plate_shape, corner_pixel=(max_pixel*0.75), max_pixel=max_pixel, well_pixel=(max_pixel*0.25))

    # Need to find the difference between the plate and the image shapes so we can
    # put the plate in the centre of the image
    x_half = int((img_shape[1] - plate_shape[1]) / 2)
    y_half = int((img_shape[0] - plate_shape[0]) / 2)

    # add the plate shell to the background array
    bkg[y_half:bkg.shape[0]-y_half,x_half:bkg.shape[1]-x_half] = plate_shell

    plt.figure()
    plt.imshow(bkg, cmap='gray')
    plt.show()

def generate_fake_plate(corner_pixel = 3500, well_pixel = 2000, max_pixel = 7000, plate_shape=(300, 400), base=False):
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
    # Cut corners off the top left and bottom left of the plate
    plate_shell = cut_plate_corner(plate_shell, corner_pixel=corner_pixel, corner="top_left")
    plate_shell = cut_plate_corner(plate_shell, corner_pixel=corner_pixel, corner="bottom_left")

    # Add evenly distributed wells
    plate_shell = add_rectangular_wells_in_plate(plate_shell, well_pixel=well_pixel)

    return plate_shell

def add_rectangular_wells_in_plate(plate_shell, well_pixel=250, n_hor = 5, n_ver = 8):
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
    ver_end_gap = plate_shape[0] / 20
    ver_spacing = int((plate_shape[0] - ver_end_gap) / n_ver)

    # calculate the column width
    well_width = int(hor_spacing - (hor_spacing / 10))
    # and well height
    well_height = int(ver_spacing - (ver_spacing / 10))

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
            well_ = galleria.well_with_galleria(well_, galleria_pixel = well_pixel * 3)
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
