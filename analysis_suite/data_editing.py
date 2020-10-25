"""
The data_editing.py file for galeria imaging

Has data editing functions
"""
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>
#

import numpy as np
import matplotlib.pyplot as plt
from json import JSONEncoder

def normalise_background_fluo(labelled_plate, fluo_image):
    """
    Takes the labelled plate image and normalises the fluo image to remove
    any natural gradient from the imaging machine (i.e. light that gets in through the door)
    """
    # get pixels outside of plate
    outside_plate = fluo_image[labelled_plate != 1]

    # we need to copy the fluo_image and set high fluo values (i.e. infected galleria)
    # to the image median as otherwise the row median ends up being high (i.e. as a result of bioluminescent galleria)
    # and we will lose all the values
    copied_plate = np.copy(fluo_image)
    copied_plate[copied_plate > 300] = np.median(outside_plate)
    # get row medians
    medians = np.median(copied_plate, axis=1)

    # divide the plate by its row median and multiply them by the whole image median
    normalised = (fluo_image / medians[:,None]) * np.median(outside_plate)
    return normalised

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class JSONEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return JSONEncoder.default(self, obj)
