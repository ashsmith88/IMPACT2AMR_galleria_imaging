"""
data_editing.py file for analysis_suite

"""

import skimage.filters as skfilt
import matplotlib.pyplot as plt
import numpy as np
import cv2

def filter_fluo_image(fluo_img, img, labelled_plate, filt=200):
    """
    Quick function to produced overlayed colormap images for Vanessa

    fluo_img : image
        fluorescence image to overlay
    img : image
        The original image
    labelled_plate : labelled image
        ndi labelled image from skimage.ndimage - where each well has a unique number starting from 2
        as the background is also labelled
    filt : int, opt
        Specific filter value to use
    """
    # convert to np array
    fluo_img = np.array(fluo_img)
    # if filt isn't specified determine it using li's threshold
    if filt is None:
        filt = skfilt.threshold_li(fluo_img)

    # create a mask for the overlay
    filtered_img = np.ma.masked_where(fluo_img < filt, fluo_img)
    # use only the wells (background is currently label 1)
    filtered_img = np.ma.masked_where(labelled_plate < 1, filtered_img)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    # plot overlayed fluo - vmax and vmin can be determined dynamically for optimal imaging
    # or hardcorded for consistency throughout tpoints (alternatively we could pass these in based on
    # thresholding across all fluo images - i.e. all tpoints)
    ax.imshow(filtered_img, cmap='jet', vmin=100, vmax=10000)#, interpolation='none')
    plt.show()
