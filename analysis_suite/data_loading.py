"""
The data_loading.py file for galeria imaging

Contains functions that load data
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>
import os
import skimage.io as skio
import analysis_suite.BR_reader.reader as biorad_reader
import re

def create_out_folder(folder):
    if os.path.isdir(os.path.join(folder, "results")):
        return os.path.join(folder, "results")
    os.mkdir(os.path.join(folder, "results"))
    return os.path.join(folder, "results")

def get_out_file(filename):
    tpoint = filename.split(" ")[1]
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
    if os.path.isfile(filepath):
        img = biorad_reader.Reader(filepath)
        loaded_image = img.get_image()
    return loaded_image

def get_image_files(folder, plate_type = "green", exposure_time = "300"):
    """
    Takes a folder and returns a list of lists, where each sublist
    is a brightfield image followed by a fluorescence image

    Parameters
    ------
    folder : str
        folder path
    plate_type : str, optional
        Specifies which plate we are analysing (default : green)
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
    files_of_interest = [file for file in os.listdir(folder) if plate_type in file]
    # extract the timepoints (assuming the filename is in the format green_t0_....)
    tpoints = [file.split(" ")[1] for file in files_of_interest]
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
            if t != file.split(" ")[1]:
                continue
            # Get brightfield images
            if "image" in file:
                tpoint_files.append(os.path.join(folder, file))
            # Get fluo images with correct exposure
            if exposure_time in file:
                tpoint_files.append(os.path.join(folder, file))
        all_files.append(sorted(tpoint_files, reverse=True))
        # Add the timepoint but drop the "t" so it is just an integer
        all_tpoints.append(int(re.search(r'\d+',t).group()))
    return all_files, all_tpoints
