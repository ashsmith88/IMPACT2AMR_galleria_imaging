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
from analysis_suite.run_yapic_model import run_model2

### TODO: importing temporarily but need to set up test running properly
import analysis_suite.tests.plate_creator as plate_creat
import analysis_suite.output as output
from analysis_suite.well_class import AllWells
import json

def run_batch(folder, plate_type):
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
        all_files, all_tpoints = load.get_image_files(folder, exposure_time='300')
        for files, tpoint in zip(all_files, all_tpoints):
            # create output folder
            out_folder = load.create_out_folder(folder)
            # analyse a tpoint brightfield and fluorescent image
            bio_dict, melanisation_dict, result_img = run_analysis(files, tpoint=tpoint, plate_type=plate_type, out_folder=out_folder)
            # add info for each bacteria to the WellData class instance
            for well, data_values in bio_dict.items():
                melanisation = melanisation_dict[well][1]
                WellData.add_well_info(well, tpoint=tpoint, area=data_values[0], mean_fluo=data_values[1], total_fluo=data_values[2], melanisation=melanisation)
            result_images[tpoint] = result_img

        # create dictionary of "plottable" dataframes where key is the info (i.e. measurement type)
        # and value is the dataframe
        WellData.create_dataframes()
        for meas, df in WellData.dataframes.items():
            WellData.dataframes[meas] = df.to_json()
        measurements_json = json.dumps(WellData.dataframes)
        images_json = json.dumps(result_images, cls=edit.NumpyArrayEncoder)
        ### # TODO: Need a save dataframe option


        #with open('measurements.json', 'w') as outfile:
            #json.dump(WellData.dataframes, outfile)
        #with open('images.json', 'w') as outfile:
        #    json.dump(result_images, outfile, cls=edit.NumpyArrayEncoder)

    else:
        ### TODO: Need to make proper error logs
        print("not a folder!")

def run_analysis(filename, plate_type, tpoint=None, out_folder=None):
    """
    Runs the main analysis pipeline

    Parameters
    ------
    filename : list
        A list of filenames to analyse
    """
    ### TODO: need to review if we want input as list or individual string
    #from timeit import default_timer as timer
    #start = timer()
    # TODO: review this as only temporary solution
    if isinstance(filename, list):
        fluo_image_file = filename[1]
        bf_image_file = filename[0]
    else:
        fluo_image_file = None
        bf_image_file = filename

    # Load the first image as a numpy array
    img = load.load_image(bf_image_file)
    img = plate_detection.straighten_plate(img)
    out_file = load.get_out_file(bf_image_file)
    # Run plate detection
    labelled_wells, labelled_plate = plate_detection.detect_plate(img, plate_type=plate_type)

    # only run for training
    #galleria_detection.save_wells_for_training(img, labelled_wells, tpoint, filename[0])

    """
    Temporary for development of model

    wells = galleria_detection.get_wells(img, labelled_wells)
    all_wells = run_model2(wells, zoom_factor=3.2)
    labelled_gall = galleria_detection.map_galleria(labelled_wells, all_wells)
    #return

    End of temporary
    """

    #well_dict, labelled_gall = galleria_detection.detect_galleria(img, labelled_wells)

    if fluo_image_file:
        # load fluo image
        fluo_image = load.load_image(fluo_image_file)

        fluo_image = edit.normalise_background_fluo(labelled_plate, fluo_image)
        # extract well data from fluo image
        ## TODO: need to extract this on galleria only?
        #bio_dict = meas.extract_biolum_values(labelled_wells, fluo_image)
        bio_dict = meas.extract_biolum_values(labelled_wells, fluo_image)
        melanisation_dict = meas.extract_melanisation_values(labelled_wells, img)

        #return bio_dict
        output.save_img(out_folder, out_file, img, labelled_plate, labelled_wells, None)
        output.save_dict(out_folder, out_file, bio_dict)
        #output.save_dict(out_folder, out_file, melanisation_dict, mel=True)

        #result_img = np.stack([img, labelled_wells, labelled_gall])
        #end = timer()
        #print(end-start, flush=True)
        return bio_dict, melanisation_dict, None #, result_img
