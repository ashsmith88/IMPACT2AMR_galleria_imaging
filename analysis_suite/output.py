"""
output.py file for analysis_suite

contains functions which export/save data
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from skimage import exposure
from skimage.color import label2rgb, gray2rgb
import skimage.io as skio
import scipy.ndimage as ndi
import analysis_suite.data_editing as edit
import json

def create_data_jsons(dataframes):
    out_dict = {}
    for meas, df in dataframes.items():
        out_dict[meas] = df.to_dict("index")
    measurements_json = json.dumps(out_dict)

    #images_json = json.dumps(result_images,cls=edit.NumpyArrayEncoder)
    ### # TODO: Need to have a different approach for images
    """
    # if we want to save them?
    with open('measurements2.json', 'w') as outfile:
        json.dump(out_dict, outfile)
    with open('images.json', 'w') as outfile:
        json.dump(result_images, outfile, cls=edit.NumpyArrayEncoder)
    """
    return measurements_json

def save_img(folder, filename, img, labelled_plate, labelled_wells, labelled_gall):
    """
    Saves an output image with the wells labelled

    Parameters
    ------
    folder : str
        Path to folder for saving
    filename : str
        Filename to save as
    img : np.array
        The original image (brightfield)
    labelled_plate : labelled image
        ndi labelled image where the plate (without the wells) is labelled as 1
    labelled_wells : labelled image
        ndi labelled image from skimage.ndimage - where each well has a unique number starting from 1
    """
    # create plot and show original image
    iter = int(5 * (img.shape[0]/2000))
    if iter == 0:
        iter = 1
    filename = filename + ".jpg"
    contours = labelled_wells * ~ndi.binary_erosion(
            labelled_wells > 0, iterations = iter)
    img = img.astype(float)
    img -= img.min()
    img /= img.max()
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    # labs = label2rgb(contours, bg_label=0)
    # imgout[contours > 0] = labs[contours > 0]
    img = (255 * gray2rgb(img)).astype('uint8')
    img[contours > 0] = (255, 255, 0)
    if labelled_gall is None:
        pilim = Image.fromarray(img)
        pilim.save(os.path.join(folder, filename))
        return
    font = ImageFont.truetype("DejaVuSans.ttf", int(56 * (img.shape[0]/2000)))
    contours_gall = labelled_gall * ~ndi.binary_erosion(
            labelled_gall > 0, iterations = iter)
    img[contours_gall > 0] = (255, 255, 0)
    pilim = Image.fromarray(img)
    pildraw = ImageDraw.ImageDraw(pilim)
    for lab in range(1, contours.max() + 1):
        coord = (contours == lab).nonzero()
        coord = [coord[0].min(), coord[1].max()]
        text = f"{lab}"
        txtsize = font.getsize(text)
        pildraw.text(
            (coord[1]-txtsize[0]-4, coord[0]-txtsize[1]),
            text,
            fill=(255, 255, 0),
            font=font)
    # skio.imsave(os.path.join(folder, filename), (255*imgout).astype('uint8'))
    pilim.save(os.path.join(folder, filename))


def save_dict(folder, filename, dictionary, mel=False):
    """
    Saves a dictionary as a csv

    Parameters
    ------
    folder : str
        Path to folder for saving
    filename : str
        Filename to save as
    dictionary : dict
        the dictionary to be saved -  keys are bacteria number and columns are measurements
        [area, fluo, integrated fluo]
    """
    ### TODO: add columns as kwarg so that the same function can be used for dictionaries
    ### with different measurements/formats

    # convert to pandas dataframe and save
    data = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Well area', 'Mean Well Fluo', 'Total Well fluo', 'Galleria area', 'Mean Galleria fluo', 'Total Galleria Fluo', 'Melanisation'])
    data.to_csv(os.path.join(folder, "%s.csv"%(filename)))
