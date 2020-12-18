# galleria_imaging

## Dependencies
The project has been tested and developed using Python 3.8.3

The main program runs with individual galleria detection which is reliant on a model developed using YAPiC.
Therefore, you will need to ensure yapic is installed and working - check the web page below for installation instructions
https://yapic.github.io/yapic/


## File naming convention
The preferred naming convention is:
Image file : tpoint_image_datestamp.tif
bioluminescent file : tpoint_exposure_datestamp.tif

Where:
tpoint is your timepoint identifier (i.e. T0)
datestamp is the experiment date (i.e. 20201016 for 16th October 2020)
exposure is the

## Running the program in batch mode

Once installed and inside the "galleria_imaging" directory. The best way to run the program is in batch mode using the command:

python -m analysis_suite path_to_folder plate_type

Where:
path_to_folder is the path to the folder containing all the data you wish to analyse (including bioluminescence images)
plate_type is the type of plate used from: "rect50", "rect40", "hex50"

Optional flag:
-exposure, -e as the exposure time of the bioluminsence image (needs to be in the file name). N.B. can be a different identifier i.e "fluo" or "bio"

## Running the program on a single file

python -m analysis_suite path_to_file plate_type

Where:
path_to_file is the path to the image you wish to analyse
plate_type is the type of plate used from: "rect50", "rect40", "hex50"

## Output

When run normally in batch mode, a "results" folder will be created in the folder that is passed in to the program.
This folder will contain a detection image and csv file for each of the timepoints analysed

## Notes on running the program for BST
When this module is run on the backend of the BST software, it will run on a separate branch where the output / inputs will
be adjusted to interact properly with the frontend software

## Production Deployment *Incomplete*

$ mkdir /var/www/galleria-imaging  
$ cd /var/www/galleria-imaging  
$ sudo -u [APP_USER] -H git clone https://github.com/ashsmith88/galleria_imaging.git code  
upload the second_model.h5 file (provided seperately due to size) to /var/www/galleria-imaging/code  
$ cd /var/www/galleria-imaging  
$ conda create --name <env_name> --file code/requirements.txt  
$ conda activate <env_name>  
$ python code/flask_api.py
