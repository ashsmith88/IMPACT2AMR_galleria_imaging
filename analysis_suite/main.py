"""
The main.py file for galeria imaging

Runs the main pipeline
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>

import analysis_suite.data_loading as load
import analysis_suite.detection as detect
import matplotlib.pyplot as plt
import os

### TODO: importing temporarily but need to set up test running properly
import analysis_suite.tests.plate_creator as plate_creat
import analysis_suite.data_loading as load
import analysis_suite.output as output

def run_batch(folder):
    """
    Takes a folder containing image files and passes them to the analysis pipeline

    Parameters
    ------
    folder : list
        List with only one entry (the batch folder)
    """
    if os.path.isdir(folder[0]):
        all_files = load.get_image_files(folder[0])
        for files in all_files:
            print(files, flush=True)
            out_folder = load.create_out_folder(folder[0])
            run_analysis(files, out_folder=out_folder)
    else:
        # TODO: Need to make proper error logs
        print("not a folder!")


def run_analysis(filename, out_folder=None):
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
    out_file = load.get_out_file(filename[0])

    # TODO: review this as only temporary solution
    if len(filename) == 2:
        fluo_image = filename[1]
    else:
        fluo_image = None

    # Run plate detection
    labelled_plate = detect.detect_plate(img)

    if fluo_image:
        fluo_image = load.load_image(fluo_image)
        labelled_plate, bio_dict = detect.extract_biolum_values(labelled_plate, fluo_image)
        output.save_img(out_folder, out_file, img, labelled_plate)
        output.save_dict(out_folder, out_file, bio_dict)



    ### # TODO: this can be removed as only used for development purposes and will
    ### Eventually only be used in test suite

    # rect ratio = 0.67, hex ratio = 0.44
    plate = plate_creat.generate_fake_plate_image(plate_ratio = 0.44)

    temp_detection_mask = plate
    temp_detection_mask[temp_detection_mask < 300] = 0
    temp_detection_mask[temp_detection_mask > 0] = 1

    from skimage.measure import regionprops
    region = regionprops(temp_detection_mask)[0]

    plate_mask = plate[region.bbox[0]: region.bbox[2], region.bbox[1] : region.bbox[3]]

    from analysis_suite.plate_dimensions import Plate
    currentplate = Plate(well_type = "hex")
    currentplate.locate_wells(plate_mask)

    """
    plt.figure()
    plt.imshow(currentplate.plate)
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(plate, cmap='gray')
    plt.show()
    """
