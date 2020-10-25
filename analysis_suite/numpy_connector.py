"""
Custom connector class that can be used to create the dataset in Yapic.

Yapic uses a TiffConnector class, however this requires tiff files as input and we
want a direct input of our numpy images
"""

import numpy as np
from yapic_io.connector import Connector
import collections

FilePair = collections.namedtuple('FilePair', ['img', 'lbl'])

class NumpyConnector(object):
    """docstring for NumpyConnector."""

    def __init__(self, images):# , label_filepath, savepath=None):

        self.images = images
        pairs = [(img, None) for img in images]

        self.filenames = [FilePair(img, Path(lbl) if lbl else None)
                          for img, lbl in pairs]

        #self.check_label_matrix_dimensions()
    def image_count(self):
        return len(self.images)

    def label_count_for_image(self, num):
        return None

    def image_dimensions(self, image_nr):
        img = self.get_correct_image(image_nr)

        Y = img[0, 0].shape[0]
        X = img[0, 0].shape[1]
        return np.hstack([1, 1, X, Y])

    def get_correct_image(self, image_nr):
        return self.images[image_nr]

    def get_tile(self, image_nr, pos, size):
        T = 0
        C, Z, X, Y = pos
        CC, ZZ, XX, YY = np.array(pos) + size

        slices = self.get_correct_image(image_nr)
        image = slices[0][0]
        tile = [[image[Y:YY, X:XX]]]
        #test2[0][Y:YY, X:XX]
        #tile = [[s[Y:YY, X:XX] for s in c[Z:ZZ]] for c in slices[T, C:CC, :]]
        tile = np.stack(tile)
        tile = np.moveaxis(tile, (0, 1, 2, 3), (0, 1, 3, 2))
        return tile.astype('float')
