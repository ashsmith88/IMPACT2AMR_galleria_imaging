"""
Tests for the galleria detection functions
"""

import unittest
import numpy as np
from analysis_suite.detection import galleria_detection
import skimage.measure as skmeas
import matplotlib.pyplot as plt
from analysis_suite.tests.galleria_creator import well_with_galleria
import scipy.ndimage as ndi

class TestDetectGalleria(unittest.TestCase):
    """
    Test function for detecting galleria in wells
    """

    def setUp(self):
        self.well_shape = (60, 100)
        self.well, self.label = well_with_galleria(np.zeros(self.well_shape), return_label=True)
        self.image_in = np.zeros((400, 200))
        self.image_out = np.zeros((400, 200))
        self.well_image = np.zeros((400, 200))
        self.out_dict = {}
        for i in range(1, 4):
            well_with_edge = np.zeros((70, 110))
            label_with_edge = np.zeros((70, 110))
            well, label = well_with_galleria(np.zeros(self.well_shape), return_label=True)
            well_with_edge[5:-5, 5:-5] = well
            label_with_edge[5:-5, 5:-5] = label
            label_with_edge[label_with_edge==1] = i
            self.out_dict[i] = label_with_edge.astype(int)
            y_start = i*100
            y_end = y_start + well_with_edge.shape[0]
            self.image_in[y_start : y_end, 50 : 160] = well_with_edge
            self.well_image[y_start : y_end, 50 : 160] = int(i)
            self.image_out[y_start : y_end, 50 : 160] = label_with_edge.astype(int)


    def test_detect_galleria_in_well(self):
        detected_gall = galleria_detection.detect_galleria_in_well(self.well)
        mismatch = (np.count_nonzero(self.label != detected_gall)) / (self.well.shape[0] * self.well.shape[1])
        self.assertTrue(mismatch < 0.01)

    def test_detect_galleria(self):
        well_dict, labelled_gall = galleria_detection.detect_galleria(self.image_in, self.well_image.astype(int))
        self.assertEqual(sorted(well_dict.keys()), sorted(self.out_dict.keys()))
        for key, val in well_dict.items():
            mismatch = (np.count_nonzero(val != self.out_dict[key])) / (val.shape[0] * val.shape[1])
            self.assertTrue(mismatch < 0.01)
        mismatch = (np.count_nonzero(labelled_gall != self.image_out)) / (labelled_gall.shape[0] * labelled_gall.shape[1])
        self.assertTrue(mismatch < 0.01)

class TestMapGalleriaPlate(unittest.TestCase):
    """
    Test function for detecting the plate
    """

    def setUp(self):
        self.galleria_dict = {1 : np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                        2 : np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                        3 : np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
        self.labelled_wells = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                                        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                                        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                                        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                                        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                                        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                                        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                                        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                                        [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.labelled_gall = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    def test_map_galleria(self):
        labelled_gall = galleria_detection.map_galleria(self.labelled_wells, self.galleria_dict)
        np.testing.assert_array_equal(self.labelled_gall, labelled_gall)

class TestFindEdgesToCrop(unittest.TestCase):
    """
    Test function for determining which edges to crop from wells
    """

    def setUp(self):
        self.well =  np.random.normal(loc=1, scale=0.3, size=(40,40)) * 100
        self.well[0:2] *= 3

        self.well1 =  np.random.normal(loc=1, scale=0.3, size=(40,40)) * 100
        self.well1[-3:] *= 3
        self.well1[:,0] *= 3

    def test_find_edges_to_crop(self):
        results = galleria_detection.find_edges_to_crop(self.well)
        self.assertEqual(results, (2, 0, 0, 0))
        results = galleria_detection.find_edges_to_crop(self.well1)
        self.assertEqual(results, (0, 3, 1, 0))


class TestCompareArrays(unittest.TestCase):
    """
    Test function for comparing arrays
    """

    def setUp(self):
        self.array1 = np.random.normal(loc=1, scale=0.7, size=20)
        self.array2 = np.random.normal(loc=2, scale=0.1, size=20)

    def test_compare_arrays(self):
        self.assertTrue(galleria_detection.compare_arrays(self.array1, self.array2))
        self.assertFalse(galleria_detection.compare_arrays(self.array2, self.array1))
        self.assertFalse(galleria_detection.compare_arrays(self.array1, self.array1))
