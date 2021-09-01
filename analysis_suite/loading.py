"""
The data_loading.py file for galeria imaging

Contains functions that load data
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>
import os
import skimage.io as skio
import analysis_suite.BR_reader.reader as biorad_reader
import re
import numpy as np
#from skimage.external import tifffile
from PIL import Image
from collections import Counter

def create_out_folder(folder):
    if os.path.isdir(os.path.join(folder, "results")):
        return os.path.join(folder, "results")
    os.mkdir(os.path.join(folder, "results"))
    return os.path.join(folder, "results")

def get_out_file(filename):
    tpoint = os.path.basename(filename).split(" ")[0]
    return "bioluminescence_reading_%s"%(tpoint)

def load_image(filepath):
    """
    Takes a filepath and loads the image

    Parameters
    ------
    filepath : str
        Path to the image file

    Returns
    ------
    img : ndarray
        The image
    """
    if filepath.lower().endswith("tif"):
        loaded_image = load_tiff_file(filepath)
    elif filepath.lower().endswith("1sc"):
        img = biorad_reader.Reader(filepath)
        loaded_image = img.get_image()
        loaded_image = np.array(loaded_image)
    return loaded_image

def load_tiff_file(filepath):
    """
    Takes a filepath and loads the tiff image

    Parameters
    ------
    filepath : str
        Path to the image file

    Returns
    ------
    img : ndarray
        The image
    """
    im = Image.open(filepath)
    return np.array(im)


def get_image_files(folder, exposure_time = "300", filetype=".tif"):
    """
    Takes a folder and returns a list of lists, where each sublist
    is a brightfield image followed by a fluorescence image

    Parameters
    ------
    folder : str
        folder path
    exposure_time : str, optional
        The exposure time we want to get the data from

    Returns
    ------
    all_files : list
        list where each entry is a list of two files ([BF_image, Fluo_image])
    all_tpoints : list
        list of timepoints as integers
    """
    # get a list of all files
    files_of_interest = [file for file in os.listdir(folder) if file.endswith(filetype)]

    # extract the timepoints (assuming the filename is in the format green_t0_....)
    tpoints = [file.split(" ")[0] for file in files_of_interest]
    # keep only unique tpoints
    tpoints = set(tpoints)

    all_files = []
    all_tpoints = []
    for t in sorted(tpoints):
        tpoint_files = []
        for file in files_of_interest:
            # if the timepoint doesn't match the file then skip
            # Need to use this rather than "in file" for instances where t = 2
            # and another time point is t=24 - ensures the whole number is in the file
            if t != file.split(" ")[0]:
                continue
            # Get brightfield images
            if "image" in file:
                tpoint_files.append(os.path.join(folder, file))
            # Get fluo images with correct exposure
            if exposure_time in file:
                tpoint_files.append(os.path.join(folder, file))
        if len(tpoint_files) == 0:
            continue
        all_files.append(sorted(tpoint_files, reverse=True))
        # Add the timepoint but drop the "t" so it is just an integer
        all_tpoints.append(int(re.search(r'\d+',t).group()))
    # We need to check they are all the same length sublists or remove them
    if len(all_files) <=1:
        return all_files, all_tpoints
    lens = Counter(len(i) for i in all_files)
    most_common_length = lens.most_common(1)[0][0]
    indices_to_remove = [n for n, x in enumerate(all_files) if len(x) != most_common_length]
    indices_to_remove = [i for i in sorted(indices_to_remove, reverse=True)]
    for ind in indices_to_remove:
        all_files.pop(ind)
        all_tpoints.pop(ind)
    return all_files, all_tpoints
