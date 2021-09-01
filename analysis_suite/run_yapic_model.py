from yapic.session import Session
from yapic_io.prediction_batch import PredictionBatch
import os
import sys
import time
import numpy as np
import tensorflow as tf
from numpy.testing import assert_equal
from bigtiff import Tiff, PlaceHolder
from pathlib import Path
from yapic_io.dataset import Dataset
import scipy.ndimage as ndi
import skimage.filters as skfilt
from skimage.measure import label
import math

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def run_model(images):
    print(tf.test.is_gpu_available())
    buff = 0
    while buff <= 1:
        try:
            with tf.device("/cpu:0"):
                tf.keras.backend.clear_session()
                converted_images, zoom_factor = convert_image_size(images, buffer=buff)
                model = "second_model.h5"
                session = Session()
                session.dataset = load_connector(converted_images)
                session.load_model(model)
                session.set_normalization('local')
                all_wells = predict(session)
                all_wells.shrink_images(zoom_factor)
            return all_wells
            break
        except:
            buff += 0.25
            continue
    return None

def convert_image_size(images, required_tile_size=317, buffer=0):
    edited_images = []
    for n, img in enumerate(images):
        if n == 0:
            x = img.shape[0]
            y = img.shape[1]
            # determine the zoom factor. Divide minimum dimension by required size.
            # Multiply by 10 and round up to nearest integer then divide by 10 so we have it up to nearest decimal place
            # Then add a buffer to be sure
            zoom_factor = (math.ceil((required_tile_size / min(x, y))  * 10) / 10) + buffer
        img = ndi.zoom(img, zoom_factor)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        edited_images.append(img)
    return edited_images, zoom_factor

def load_connector(images):
    """
    Connect to a prediction dataset.
    Parameters
    ----------
    images : list
        list of np.arrays which are well images
    """

    from .numpy_connector import NumpyConnector
    dataset = Dataset(NumpyConnector(images))
    return dataset

def predict(session):
    data_predict = PredictionBatch(session.dataset,
                                   2,
                                   session.output_tile_size_zxy,
                                   session.padding_zxy)
    data_predict.set_normalize_mode('local')
    data_predict.set_pixel_dimension_order('bzxyc')

    print("Starting predictions...")
    start = time.time()
    all_wells = AllWells()
    for n, item in enumerate(data_predict, start=1):
        print("Processing image %s of %s"%(n, len(data_predict)))
        start = time.time()

        result = session.model.predict(item.pixels())

        pred = time.time()
        print("Predicted in %s seconds"%(pred-start))
        #item.put_probmap_data(result)
        tiles, x_axis, y_axis, image_num, positions = put_probmap_data(item, result)
        all_wells.add_tile_to_well(image_num+1, tiles, positions, x_axis, y_axis)
        end = time.time()
        print("Done in %s seconds"%(end-start))
    all_wells.create_labelled_images()
    return all_wells
    end = time.time()
    #print("Done all predictions in %s seconds"%(end-start))

class AllWells(object):
    """docstring for AllWells."""

    def __init__(self):
        super(AllWells, self).__init__()

        self._individual_wells = {}

    def add_tile_to_well(self, well_num, tiles, positions, x_axis, y_axis):
        if well_num in self._individual_wells.keys():
            self.update_well_data(well_num, tiles, positions, x_axis, y_axis)
        else:
            self.add_well_data(well_num, tiles, positions, x_axis, y_axis)

    def add_well_data(self, well_num, tiles, positions, x_axis, y_axis):
        self._individual_wells[well_num] = TileCompiler(well_num)
        self._individual_wells[well_num].add_new_tile(tiles, positions, x_axis, y_axis)

    def update_well_data(self, well_num, tiles, positions, x_axis, y_axis):
        self._individual_wells[well_num].add_new_tile(tiles, positions, x_axis, y_axis)

    def create_labelled_images(self):
        for well in self._individual_wells.values():
            well.combine_tiles()

    def shrink_images(self, factor):
        for well in self._individual_wells.values():
            well.shrink_image(factor)


class TileCompiler(object):
    """docstring for TileCompiler."""

    def __init__(self, well_num):
        super(TileCompiler, self).__init__()
        self.well_num = well_num
        self.max_x = 0
        self.max_y = 0
        self.tiles = []
        self.output_array = None # currently this is set as the output but need to turn into label mask

    def shrink_image(self, factor):
        self.output_array = ndi.zoom(self.output_array, (1 / factor))
        self.output_array = ndi.morphology.binary_fill_holes(self.output_array) + 0

    def add_new_tile(self, tiles, positions, max_x, max_y):
        # Get the max x and y values in order to create the final array later
        if max_x > self.max_x:
            self.max_x = max_x
        if max_y > self.max_y:
            self.max_y = max_y
        # add the tile info
        # # TODO:  at the moment there are 2 tiles in each result - may change with new model so keep this
        # in mind!
        if len(tiles) == 1:
            self.tiles.append([positions, tiles])
        else:
            for pos, tile in zip(positions, tiles):
                self.tiles.append([pos, tile])

    def combine_tiles(self):
        img_array = np.zeros((self.max_y,self.max_x))
        for tile_info in self.tiles:
            tile = tile_info[1]
            y_start = tile_info[0][1]
            x_start = tile_info[0][0]
            img_array[y_start:(y_start+tile.shape[0]), x_start:(x_start+tile.shape[1])] = tile

        img_array = self.convert_to_label(img_array)

        self.output_array = img_array

    def convert_to_label(self, img):
        filt = skfilt.threshold_otsu(img)
        label_img = np.zeros(img.shape)
        label_img[img < filt] = 1
        #label_img = ndi.label(label_img)[0]
        label_img = self.getLargestlabel(label_img)
        return label_img

    def getLargestlabel(self, segmentation):
        labels = label(segmentation)
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC

def put_probmap_data(item, probmap_batch):
        probmap_batch = np.moveaxis(probmap_batch,
                                    item.pixel_dimension_order,
                                    [0, 1, 2, 3, 4])

        assert_equal(len(probmap_batch.shape), 5, '5-dim (B,L,Z,X,Y) expected')
        B, L, *ZXY = probmap_batch.shape
        item.labels = np.arange(L) + 1 if len(item.labels) == 0 \
            else item.labels

        assert_equal(B, len(item.current_tile_positions))
        assert_equal(L, len(item.labels))
        assert_equal(ZXY, item.tile_size_zxy)

        tiles = []
        x_vals = []
        y_vals = []
        image_num = 0
        positions = []
        for probmap, (image_nr, pos_zxy) in zip(probmap_batch,
                                                item.current_tile_positions):
            tile = get_tile(item, probmap[0], pos_zxy, image_nr, item.labels[0])
            x, y = get_array_shape(probmap, image_nr, pos_zxy)
            x_vals.append(x)
            y_vals.append(y)
            tiles.append(tile)
            image_num = image_nr
            positions.append(pos_zxy[1:])
        return tiles, max(x_vals), max(y_vals), image_nr, positions

def get_array_shape(pixels, image_nr, pos_zxy):
    pixels = np.array(pixels[0], dtype=np.float32)
    ZZ, XX, YY = np.array(pos_zxy) + pixels.shape
    return XX, YY

def _open_probability_map_file(item, image_nr, label_value):
    # memmap is slow, so we must cache it to be fast!
    fname = item.dataset.pixel_connector.filenames[image_nr].img
    fname = Path('{}_class_{}.tif'.format(fname.stem, label_value))

    path = item.dataset.pixel_connector.savepath / fname

    if not path.exists():
        T = 1
        C = 1
        _, Z, X, Y = item.dataset.pixel_connector.image_dimensions(image_nr)
        images = [PlaceHolder((Y, X, 1), 'float32')] * Z
        Tiff.write(images, io=str(path), imagej_shape=(T, C, Z))

    return Tiff.memmap_tcz(path)

def get_tile(item, pixels, pos_zxy, image_nr, label_value):
    #assert self.savepath is not None
    np.testing.assert_equal(len(pos_zxy), 3)
    np.testing.assert_equal(len(pixels.shape), 3)
    pixels = np.array(pixels, dtype=np.float32)

    tile = pixels[0, :, :].T

    return tile
    """
    slices = _open_probability_map_file(item, image_nr, label_value)

    T = C = 0
    Z, X, Y = pos_zxy
    ZZ, XX, YY = np.array(pos_zxy) + pixels.shape

    print(Z, X, Y)
    print(pixels.shape)
    print(ZZ, XX, YY)

    for z in range(Z, ZZ):
        slices[T, C, z][Y:YY, X:XX] = pixels[z - Z, ...].T
    """


def get_file_list(in_folder):
    files = [os.path.join(in_folder, file) for file in os.listdir(in_folder) if file.endswith('.tif')]
    return files

"""
if __name__ == '__main__':
    model = sys.argv[1]
    in_folder = sys.argv[2]
    out_folder = sys.argv[3]
    run_model(model, in_folder, out_folder)
"""

if __name__ == '__main__':
    model = sys.argv[1]
    in_folder = sys.argv[2]
    out_folder = sys.argv[3]
    run_model(model, in_folder, out_folder)
