# galleria_imaging


## File naming convention
For the brightfield images, each image should contain a timepoint identifier and the word "image". For example:
t0_image.tif
If you are including a bioluminescent file, then the same timepoint naming convention should be used, but you should change the word "image" for an identifier. This could be the exposure time (e.g. "300") or simply a tag (e.g. "bio"). For example:
t0_bio.tif
This tag needs to be passed into the analysis, either using the flag -exposure if using the CLI, or entered into the Graphical User Interface (GUI).


# Installation 

## Setting up python

Installing the Galleria Imaging analysis suite is relatively straightforward. The first step is to ensure you have python installed and working on your machine. 
The installation method will vary depending on your operating system but there are plenty of tutorials / instructions available with a simple google search. 

## Downloading the repository

Once you have python installed. You need to copy this repository onto your local system. This can be done using one simple command from inside your terminal window:

```git clone git@github.com:ashsmith88/galleria_imaging.git```

Alternatively, by clicking on the green "code" button at the top of this page, there is the option to download the folder as a zip.

Once download, you need to move in to the folder in your terminal. This is done using the ```cd``` command:

```cd /path/to/galleria_imaging```

## Installing the required modules 

### Setting up a Virtual Environment (optional)

If you plan on using python for other projects, it can be good to set up a virtual environment to ensure you don't have conflicts with dependencies at a future date.
If you are just going to be using it for this program, you can avoid this step.
A virtual environemnt can be created using the command:

```python -m venv venv```

We then must activate the venv (note, if using a venv, you will need run this following command whenever you want to run the program. 

```source activate venv/bin/activate```

### Running with GPU

The detection of the galleria relies on a module called YAPiC and it runs MUCH quicker if you have the option of using a GPU. 
The steps for installing and activating this vary depending on your computer so you may need to look into how to ensure it is set up properly. 

Some useful links for ensuring tensorflow, CUDA, etc. are correctly installed can be found on the YAPiC website:
https://yapic.github.io/yapic/


### Installation 

If you are a linux user, you can install the program by simply running:

```make init```

If this doesn't work, or you are on a different Operating system, you can run the same installation steps using these commands:

```pip install --upgrade pip setuptools```
```pip install -r requirements.txt```

# Graphical User Interface (GUI)

If you are a linux user you can initiate the GPU by running:

```make run```

If this doesn't work, or you are on a different Operating system, you can run the same installation steps using these commands:

```python -m analysis_suite.gui```

When the GUI has started, you can use the button to search for the folder containing your images, or you can simply drag and drop the folder onto the GUI. 

Note, the GUI only runs the software in batch mode, so ensure all your images are in a single folder and follow the above naming convention. 

Once you have seleced or dragged and dropped your folder, you can enter your unique identifier for your bioluminescent files, then click the "Start Analysis" button.

You will probably notice some text being printed in your terminal, don't worry about this. You now just need to wait!

The speed varies massively depending on both your individual machine, and if you have GPU activated. However, a rough estimate, is that using GPU, a single image should be analysed in in a minute. However, on CPU this is a lot slower and you can be waiting for 5+minutes for each image. Be Patient!

Once it has finished, the "Open results folder" will become enabled and you can simply click on this to open a file explorer window where you should now find a results folder in your image folder!

# CLI

## Running the program in batch mode

Once installed and inside the "galleria_imaging" directory. The best way to run the program is in batch mode using the command:

python -m analysis_suite path_to_folder -batch

Where:
path_to_folder is the path to the folder containing all the data you wish to analyse (including bioluminescence images)

Optional flag:
-exposure, -e as the exposure time of the bioluminsence image (needs to be in the file name). N.B. can be a different identifier i.e "fluo" or "bio"

## Running the program on a single file

python -m analysis_suite path_to_file

Where:
path_to_file is the path to the image you wish to analyse

Note that this will only run on a brightfield image

## Output

When run normally in batch mode, a "results" folder will be created in the folder that is passed in to the program.
This folder will contain a detection image and csv file for each of the timepoints analysed
