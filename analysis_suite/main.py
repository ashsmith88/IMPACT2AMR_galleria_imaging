"""
The main.py file for galeria imaging

Runs the main pipeline
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>

import analysis_suite.data_loading as load
import matplotlib.pyplot as plt
import numpy as np

def run_analysis(filename):
    """
    Runs the main analysis pipeline

    Parameters
    ------
    filename : list
        A list of filenames to analyse
    """
    ### TODO: need to review if we want input as list or individual string

    # Load the first image
    img = load.load_image(filename[0])
    generate_fake_plate_image()


def generate_fake_plate_image(max_bkg = 250, max_pixel = 7000, img_shape=(346, 464), plate_shape=(300, 400)):
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

    bkg = np.random.randint(max_bkg, size=img_shape)

    plate_shell = generate_fake_plate(plate_shape=plate_shape)
    x_half = int((img_shape[1] - plate_shape[1]) / 2)
    y_half = int((img_shape[0] - plate_shape[0]) / 2)

    bkg[y_half:bkg.shape[0]-y_half,x_half:bkg.shape[1]-x_half] = plate_shell

    plt.figure()
    plt.imshow(bkg)
    plt.show()

def generate_fake_plate(max_bkg = 250, max_pixel = 7000, plate_shape=(300, 400)):
    """
    Generates a fake plate with given dimensions and corner cut offs in two corners

    Parameters
    ------
    max_bkg : int
        Maximum pixel intensity for the background
    max_pixel : int
        Average maximum pixel used for gaussian randomly generate pixel values
    plate_shape : tuple
        Rough external dimensions of plate to be created
    """
    plate_shell = (np.random.normal(loc=1, scale=0.05 , size=plate_shape)*7000)
    plate_shell = cut_plate_corner(plate_shell)
    plate_shell = cut_plate_corner(plate_shell, corner="bottom_left")

    plate_shell = add_rectangular_wells_in_plate(plate_shell, max_bkg=max_bkg)

    return plate_shell

def add_rectangular_wells_in_plate(plate_shell, max_bkg=250, n_hor = 5, n_ver = 8):
    """
    Adds rectangular wells in to the fake plate

    Parameters
    ------
    plate_shell : ndarry
        2D numpy array in which the wells need to be added
    max_bkg : int
        Maximum pixel intensity for the background
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

    x_row_starts = []
    current_x = hor_end_gap
    while current_x <= (plate_shape[1] - hor_spacing):
        x_row_starts.append(int(current_x))
        current_x = current_x + hor_spacing

    y_row_starts = []
    current_y = ver_end_gap
    while current_y <= (plate_shape[0] - ver_spacing):
        y_row_starts.append(int(current_y))
        current_y = current_y + ver_spacing

    for x_val in x_row_starts:
        for y_val in y_row_starts:
            well_ = np.random.randint(max_bkg, size=(well_height, well_width))
            plate_shell[y_val:y_val+well_height, x_val:x_val+well_width] = well_

    return plate_shell

def cut_plate_corner(plate_shell, max_bkg = 250, num_pixels=15, corner="top_left"):
    """
    "Cuts" the corner of a numpy array by setting triangles in the corner to
    random integers up to the maximum specified

    Parameters
    ------

    plate_shell : ndarry
        2D numpy array in which the corners need to be "cut"
    max_bkg : int
        Maximum pixel intensity for the background
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

    coords = []
    for n, x_val in enumerate(x_vals, start=0):
        for y_val in y_vals[:(len(y_vals) - n)]:
            coords.append((y_val, x_val))

    for coo_ in coords:
        plate_shell[coo_] = np.random.randint(max_bkg)

    return plate_shell
