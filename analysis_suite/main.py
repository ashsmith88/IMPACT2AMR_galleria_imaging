"""
The main.py file for galeria imaging

Runs the main pipeline
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>

import analysis_suite.data_loading as load

### TODO: importing temporarily but need to set up test running properly
import analysis_suite.tests.plate_creator as plate
import analysis_suite.tests.galleria_creator as galleria

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
    plate.generate_fake_plate_image()
    #galleria.well_with_galleria()
