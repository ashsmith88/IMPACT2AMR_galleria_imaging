"""
Tests for the detection functions
"""

import unittest
import numpy as np
from analysis_suite import detection as detect
from analysis_suite.tests.plate_creator import generate_fake_plate_image

class TestExtractBiolumValues(unittest.TestCase):
    """
    Test function for extracting biolumninescence values
    """

    def setUp(self):
        self.labelled_wells = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0],
                                        [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0],
                                        [0, 0, 0, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 6, 6, 0, 7, 7, 0, 8, 8, 0, 9, 9, 0, 10, 10, 0],
                                        [0, 6, 6, 0, 7, 7, 0, 8, 8, 0, 9, 9, 0, 10, 10, 0],
                                        [0, 6, 6, 0, 7, 7, 0, 0, 0, 0, 9, 9, 0, 10, 10, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                                        )

        self.fluo_image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 100, 200, 0, 200, 200, 0, 300, 100, 0, 400, 450, 0, 500, 520, 0],
                                        [0, 200, 100, 0, 210, 210, 0, 300, 200, 0, 450, 400, 0, 540, 560, 0],
                                        [0, 0, 0, 0, 220, 220, 0, 300, 300, 0, 500, 500, 0, 580, 600, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 600, 500, 0, 100, 100, 0, 400, 800, 0, 100, 100, 0, 101, 102, 0],
                                        [0, 100, 200, 0, 94, 100, 0, 800, 800, 0, 100, 100, 0, 104, 103, 0],
                                        [0, 400, 300, 0, 100, 100, 0, 0, 0, 0, 100, 160, 0, 105, 103, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                                        )

        self.output_dict = {1 : [4, 150, 600],
                            2 : [6, 210, 1260],
                            3 : [6, 250, 1500],
                            4 : [6, 450, 2700],
                            5 : [6, 550, 3300],
                            6 : [6, 350, 2100],
                            7 : [6, 99, 594],
                            8 : [4, 700, 2800],
                            9 : [6, 110, 660],
                            10 : [6, 103, 618]}

    def test_extract_biolum_values(self):
        out_dict = detect.extract_biolum_values(self.labelled_wells, self.fluo_image)
        self.assertEqual(out_dict, self.output_dict)

class TestExtractProfiles(unittest.TestCase):
    """
    Test function for identifying the first wells
    """

    def setUp(self):
        self.profile = np.random.randint(low=0, high=100, size=300)
        peak_positions = [2, 3, 70, 71, 72, 73, 74, 75, 98, 99, 100, 124, 125, 148, 149, 173, 174, 198, 199, 200, 222, 223, 224, 248, 249]
        for peak in peak_positions:
            self.profile[peak] = np.random.randint(low=2000, high=2500)
        self.profile_diff = np.diff(self.profile)
        self.first_peak = 75
        self.median_gap = 25

    def test_find_first_well(self):
        peaks, gap = detect.find_first_well(self.profile_diff, num_peaks = 8)
        self.assertEqual(peaks, self.first_peak)
        self.assertEqual(self.median_gap, gap)


class TestGetFirstWellAndGaps(unittest.TestCase):
    """
    Test function for finding first the first well and the well spacing
    """

    def setUp(self):
        self.img50_hex = generate_fake_plate_image(plate_length = 180, plate_ratio = 0.4, hex=True, n_ver=10)
        self.img40_well = generate_fake_plate_image(plate_length = 280, plate_ratio = 0.67)
        self.img50_well = generate_fake_plate_image(plate_length = 280, plate_ratio = 1, n_ver = 10)
    """
    def test_get_first_well_and_gaps_40_well(self):
        start_x, start_y, x_gap, y_gap = detect.get_first_well_and_gaps(self.img40_well, 8, 5)
        self.assertTrue(62 <= start_x <= 64)
        self.assertTrue(38 <= start_y <= 40)
        self.assertTrue(75 <= x_gap <= 77)
        self.assertTrue(33 <= y_gap <= 35)

    def test_get_first_well_and_gaps_50_well(self):
        start_x, start_y, x_gap, y_gap = detect.get_first_well_and_gaps(self.img50_well, 10, 5)
        self.assertTrue(118 <= start_x <= 120)
        self.assertTrue(38 <= start_y <= 40)
        self.assertTrue(50 <= x_gap <= 52)
        self.assertTrue(27 <= y_gap <= 29)
    """
    def test_get_first_well_and_gaps_50_hex(self):
        start_x, start_y, x_gap, y_gap = detect.get_first_well_and_gaps(self.img50_hex, 10, 5)
        print(start_x, start_y, x_gap, y_gap, flush=True)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.img50_hex)
        plt.show()
