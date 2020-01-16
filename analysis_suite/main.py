"""
The main.py file for galeria imaging

Runs the main pipeline
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>

import analysis_suite.data_loading as load
import analysis_suite.detection.plate_detection as plate_detection
import analysis_suite.measurements as meas
import matplotlib.pyplot as plt
import os

### TODO: importing temporarily but need to set up test running properly
import analysis_suite.tests.plate_creator as plate_creat
import analysis_suite.data_loading as load
import analysis_suite.output as output
import analysis_suite.plotting as plotting
from analysis_suite.well_class import AllWells
import analysis_suite.data_editing as edit

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

    if os.path.isdir(folder):
        # get a list of files and timepoints
        all_files, all_tpoints = load.get_image_files(folder, exposure_time='300')
        for files, tpoint in zip(all_files, all_tpoints):
            # create output folder
            out_folder = load.create_out_folder(folder)
            # analyse a tpoint brightfield and fluorescent image
            bio_dict = run_analysis(files, plate_type=plate_type, out_folder=out_folder)
            # add info for each bacteria to the WellData class instance
            for well, data_values in bio_dict.items():
                WellData.add_well_info(well, tpoint=tpoint, area=data_values[0], mean_fluo=data_values[1], total_fluo=data_values[2])
        # create dictionary of "plottable" dataframes where key is the info (i.e. measurement type)
        # and value is the dataframe
        WellData.create_dataframes()
        for info, data in WellData.dataframes.items():
            # plot dataframe of choice
            plotting.plot_interactive_chart(data)
            plt.close()
            #plt.show()
    else:
        ### TODO: Need to make proper error logs
        print("not a folder!")




def run_analysis(filename, plate_type, out_folder=None):
    """
    Runs the main analysis pipeline

    Parameters
    ------
    filename : list
        A list of filenames to analyse
    """
    ### TODO: need to review if we want input as list or individual string

    # TODO: review this as only temporary solution
    if isinstance(filename, list):
        fluo_image_file = filename[1]
        bf_image_file = filename[0]
    else:
        fluo_image_file = None
        bf_image_file = filename

    # Load the first image
    img = load.load_image(bf_image_file)
    out_file = load.get_out_file(bf_image_file)
    # Run plate detection
    labelled_wells, labelled_plate = plate_detection.detect_plate(img, plate_type=plate_type)

    if fluo_image_file:
        # load fluo image
        fluo_image = load.load_image(fluo_image_file)
        # extract well data from fluo image
        bio_dict = meas.extract_biolum_values(labelled_wells, fluo_image)
        #edit.filter_fluo_image(fluo_image, img, labelled_plate)
        output.save_img(out_folder, out_file, img, labelled_plate, labelled_wells)
        output.save_dict(out_folder, out_file, bio_dict)
        return bio_dict
