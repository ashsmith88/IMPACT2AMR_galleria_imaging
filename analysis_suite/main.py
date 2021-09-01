"""
The main.py file for galeria imaging

Runs the main pipeline
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>

import analysis_suite.loading as load
import analysis_suite.detection.plate_detection as plate_detection
import analysis_suite.detection.galleria_detection as galleria_detection
import analysis_suite.measurements as meas
import analysis_suite.data_editing as edit
import matplotlib.pyplot as plt
import os
import numpy as np
from analysis_suite.run_yapic_model import run_model

### TODO: importing temporarily but need to set up test running properly
import analysis_suite.tests.plate_creator as plate_creat
import analysis_suite.output as output
from analysis_suite.well_class import AllWells


def run_batch(folder, plate_type="rect50", exposure='300', from_gui=False):
    """
    Takes a folder containing image files and passes them to the analysis pipeline

    Parameters
    ------
    folder : list
        List with only one entry (the batch folder)
    """

    # Create an instance of AllWells class
    WellData = AllWells()

    result_images = {}
    if os.path.isdir(folder):
        # get a list of files and timepoints
        all_files, all_tpoints = load.get_image_files(folder, exposure_time=exposure)
        for files, tpoint in zip(all_files, all_tpoints):
            # create output folder
            out_folder = load.create_out_folder(folder)
            # analyse a tpoint brightfield and fluorescent image
            melanisation_dict, bio_dict_wells, bio_dict_gall = run_analysis(files, plate_type=plate_type, out_folder=out_folder)
            if bio_dict_wells is None or melanisation_dict is None or bio_dict_gall is None:
                # TODO: add error message
                continue
            # add info for each bacteria to the WellData class instance
            for well, data_values in bio_dict_wells.items():
                melanisation = melanisation_dict[well][1]
                gall_data = bio_dict_gall[well]
                WellData.add_well_info(well, tpoint=tpoint, area_well=data_values[0], mean_fluo_well=data_values[1], total_fluo_well=data_values[2], melanisation=melanisation,
                                    area_gall=gall_data[0], mean_fluo_gall=gall_data[1], total_fluo_gall=gall_data[2])

        # create dictionary of "plottable" dataframes where key is the info (i.e. measurement type)
        # and value is the dataframe
        WellData.create_dataframes()
        measurements_json = output.create_data_jsons(WellData.dataframes, out_folder=out_folder)
        if from_gui:
            return out_folder
        return WellData.dataframes
    else:
        ### TODO: Need to make proper error logs
        print("not a folder!")

def run_analysis(filename, plate_type=None, out_folder=None):
    """
    Runs the main analysis pipeline

    Parameters
    ------
    filename : list
        A list of filenames to analyse
    """
    if len(filename) == 2:
        fluo_image_file = filename[1]
        bf_image_file = filename[0]
    elif isinstance(filename, list):
        fluo_image_file = None
        bf_image_file = filename[0]
    elif isinstance(filename, str):
        fluo_image_file = None
        bf_image_file = filename

    # Load the first image as a numpy array
    img = load.load_image(bf_image_file)
    if img.ndim == 3: # convert it to a greyscale uint16 image
        img = ((img[:,:,0] / 255) * 65535).astype("uint16")

    angle = plate_detection.detect_plate_rotation(img)
    img = plate_detection.straighten_plate(img, angle)

    out_file = load.get_out_file(bf_image_file)
    # Run plate detection
    labelled_wells, labelled_plate = plate_detection.detect_plate(img, plate_type=plate_type)
    if labelled_wells is None or labelled_plate is None:
        ## TODO:  add error message
        return None, None, None
    labelled_gall = None

    wells = galleria_detection.get_wells(img, labelled_wells)
    all_wells = run_model(wells)
    if all_wells is not None:
        labelled_gall = galleria_detection.map_galleria(labelled_wells, all_wells)

    if fluo_image_file:
        # load fluo image
        fluo_image = load.load_image(fluo_image_file)
        if fluo_image.ndim == 3: # convert it to a greyscale uint16 image
            fluo_image = ((fluo_image[:,:,0] / 255) * 65535).astype("uint16")

        fluo_image = plate_detection.straighten_plate(fluo_image, angle)

        fluo_image = edit.normalise_background_fluo(labelled_plate, fluo_image)
    else:
        fluo_image = None

    bio_dict_wells = meas.extract_biolum_values(labelled_wells, fluo_image)
    if labelled_gall is not None:
        bio_dict_gall = meas.extract_biolum_values(labelled_gall, fluo_image)
        melanisation_dict = meas.extract_melanisation_values(labelled_gall, img)
    else:
        bio_dict_gall = bio_dict_wells
        melanisation_dict = meas.extract_melanisation_values(labelled_wells, img)

    all_data = {}
    for tpoint, data in bio_dict_wells.items():
        tpoint_data = data + bio_dict_gall[tpoint] + [melanisation_dict[tpoint][1]]
        all_data[tpoint] = tpoint_data

    output.save_img(out_folder, out_file, img, labelled_plate, labelled_wells, labelled_gall)
    output.save_dict(out_folder, out_file, all_data)
    return melanisation_dict, bio_dict_wells, bio_dict_gall
