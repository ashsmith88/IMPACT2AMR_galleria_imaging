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


def generate_fake_plate_image(max_pixel = 7000, img_shape=(364, 464), plate_shape=(300, 400)):
    """
    Generates a fake image with a plate that can be used to test detection

    Parameters
    ------
    max_pixel : int
        Average maximum pixel used for gaussian randomly generate pixel values
    img_shape : tuple
        Tuple containing the dimensions of the image to be created
    """

    bkg = np.random.randint(250, size=img_shape)

    plate_shell = (np.random.normal(loc=1, scale=0.05 , size=plate_shape)*7000)

    plate_shell.max()

    bkg[32:bkg.shape[0]-32,32:bkg.shape[1]-32] = plate_shell

    plt.figure()
    plt.imshow(bkg)
    plt.show()
