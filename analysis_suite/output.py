"""
output.py file for analysis_suite

contains functions which export/save data
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def save_img(folder, filename, img, labelled_plate):
    """
    Saves an output image with the wells labelled

    Parameters
    ------
    folder : str
        Path to folder for saving
    filename : str
        Filename to save as
    img : np.array
        The original image (brightfield)
    labelled_plate : labelled image
        ndi labelled image from skimage.ndimage - where each well has a unique number starting from 2
        as the background is also labelled
    """
    # create plot and show original image
    plt.figure()
    plt.imshow(img)
    # loop through labelled image and plot the contours of each label
    for well_lab in range(1, labelled_plate.max() + 1):
        # create colour for contour
        col = plt.cm.gist_rainbow((well_lab / 9.1) % 1)
        # plot contour
        plt.contour(labelled_plate == well_lab, levels=[0.5], colors=[col])
        # get the location of the well and find the top right corner in order to add the text label
        bw0 = labelled_plate == well_lab
        pos0 = bw0.nonzero()
        pos = (np.min(pos0[0]), np.max(pos0[1]))
        plt.text(pos[1], pos[0], str(well_lab), color=col)
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def save_dict(folder, filename, dictionary):
    """
    Saves a dictionary as a csv

    Parameters
    ------
    folder : str
        Path to folder for saving
    filename : str
        Filename to save as
    dictionary : dict
        the dictionary to be saved -  keys are bacteria number and columns are measurements
        [area, fluo, integrated fluo]
    """
    ### TODO: add columns as kwarg so that the same function can be used for dictionaries
    ### with different measurements/formats

    # convert to pandas dataframe and save
    data = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Area', 'Mean Fluo', 'Total fluo'])
    data.to_csv(os.path.join(folder, "%s.csv"%(filename)))
