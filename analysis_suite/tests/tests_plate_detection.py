"""
Tests for the detection functions
"""

import unittest
import numpy as np
from analysis_suite.detection import plate_detection
from analysis_suite.tests.plate_creator import generate_fake_plate_image
import analysis_suite.tests.galleria_creator as galleria
import skimage.measure as skmeas
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

class TestRotatePlate(unittest.TestCase):
    """
    Tests the plate rotation functions
    """
    def setUp(self):
        self.img50_well = generate_fake_plate_image(plate_length = 280, plate_ratio = 0.96, n_ver = 10)
        self.angles = [-3, -2, -1, 1, 2, 3]

    def test_detect_plate_rotation(self):
        for angle in self.angles:
            rotated_plate = ndi.rotate(self.img50_well, angle, mode='reflect')
            detected_angle = plate_detection.detect_plate_rotation(rotated_plate)
            self.assertTrue(-(angle+0.25) <= detected_angle <= -(angle-0.25))

class TestResizeImage(unittest.TestCase):
    """
    Test function for resizing the image
    """

    def setUp(self):
        self.image_400 = np.zeros((400, 400))
        self.image_1600 = np.zeros((1600, 1600))
        self.image_200 = np.zeros((200, 200))

    def test_resize_image_no_change(self):
        image = plate_detection.resize_image(self.image_400)
        np.testing.assert_array_equal(image, self.image_400)

    def test_resize_image_no_change(self):
        image = plate_detection.resize_image(self.image_1600)
        np.testing.assert_array_equal(image, self.image_400)

    def test_resize_image_small(self):
        image = plate_detection.resize_image(self.image_200)
        np.testing.assert_array_equal(image, self.image_200)

class TestAdjustAngle(unittest.TestCase):
    """
    Test function for getting the adjusted angles
    """

    def setUp(self):
        self.in_angles = [0, 35, 88, 182, 270, 357]
        self.out_angles = [0, False, -2, 2, 0, -3]

    def test_adjust_angles(self):
        for in_ang, out_ang in zip(self.in_angles, self.out_angles):
            new_angle = plate_detection.adjust_angle(in_ang)
            self.assertEqual(out_ang, new_angle)


class TestDetectPlate(unittest.TestCase):
    """
    Test function for detecting the plate
    """

    def setUp(self):
        self.img_shape = ((346, 464))
        self.hex_plate_50 = {"plate_dim":(85, 195), "well_dim":(17.5, 15), "first_well":(5, 12.92),
            "col_space" : 17, "row_space" : 29, "row_space_no_stagger" : 14.75, "nrows" : 10, "ncols" : 5} ## TODO: need to measure row space no stagger
        self.rect_plate_40 = {"plate_dim":(172, 254), "well_dim":(16, 40), "first_well":(8, 21),
            "col_space" : 45, "row_space" : 20, "row_space_no_stagger" : 20, "nrows" : 8, "ncols" : 5}
        self.rect_plate_50 = {"plate_dim":(270, 280), "well_dim":(18, 42), "first_well":(10, 25),
            "col_space" : 50, "row_space" : 26, "row_space_no_stagger" : 26, "nrows" : 10, "ncols" : 5}

    def rect_plate_maker(self, dict, just_labels=False):
        # if just labels then we need 0s, whereas for the plate that is going to be detected
        # we need to have random values in given ranges
        if just_labels:
            plate = np.zeros(dict["plate_dim"])
        else:
            plate = (np.random.normal(loc=1, scale=0.05 , size=dict["plate_dim"])*25000)
        start_y = int(dict["first_well"][0])
        start_x = int(dict["first_well"][1])
        x_gap = int(dict["col_space"])
        y_gap = int(dict["row_space"])
        # Lets loop through the rows and columns and stamp the wells
        for row in range(0, dict["nrows"]):
            for col in range(0, dict["ncols"]):
                if just_labels:
                    well = np.zeros((int(dict["well_dim"][0]), int(dict["well_dim"][1]))) + 1
                else:
                    well = np.random.normal(loc=1, scale=0.05, size=dict["well_dim"])*7000
                x_coo = start_x + (x_gap * col)
                y_coo = start_y + (y_gap * row)
                if not just_labels:
                    well = galleria.well_with_galleria(well, galleria_pixel = 25000)
                # "stamp" the well in place
                plate[y_coo:(y_coo + int(dict["well_dim"][0])), x_coo:(x_coo + int(dict["well_dim"][1]))] = well
        return plate

    def add_plate_to_image(self, image, plate):
        """
        Simply adds a plate to the centre of an image
        """
        # Need to find the difference between the plate and the image shapes so we can
        # put the plate in the centre of the image
        x_half = int((self.img_shape[1] - plate.shape[1]) / 2)
        y_half = int((self.img_shape[0] - plate.shape[0]) / 2)
        # add the plate shell to the background array
        image[y_half:y_half+plate.shape[0],x_half:x_half+plate.shape[1]]  = plate
        return image

    def test_detect_img_40_well(self):
        image = np.random.randint(7000, size=self.img_shape)
        self.run_test(image, self.rect_plate_40, plate_type="rect40")

    def test_detect_img_50_well(self):
        image = np.random.randint(7000, size=self.img_shape)
        self.run_test(image, self.rect_plate_50, plate_type="rect50")


    def run_test(self, image, plate_dict, plate_type="rect40"):
        """
        This function simply builds the fake plate and respective labelled plate and labelled well
        images then tests the mismatch between these labelled images is less than 1% for the detected plate
        and less than 5% for the detected wells (wells are slightly higher as less of the image is labelled)
        The reason we can't test they match exactly is because the plate dimensions aren't integers whereas
        slicing/indexing array requires integers.
        """
        filled_plate = self.rect_plate_maker(plate_dict)
        image = self.add_plate_to_image(image, filled_plate)
        blank_image = np.zeros(image.shape)
        labelled_plate = self.add_plate_to_image(np.zeros(image.shape), (np.zeros(filled_plate.shape)+1))
        labelled_wells = self.rect_plate_maker(plate_dict, just_labels=True)
        labelled_wells = self.add_plate_to_image(blank_image, labelled_wells)
        labelled_wells = skmeas.label(labelled_wells, return_num=False)
        detected_wells, detected_plate = plate_detection.detect_plate(image, plate_type=plate_type)
        mismatch_plate = (np.count_nonzero(labelled_plate != detected_plate)) / (image.shape[0] * image.shape[1])
        mismatch_wells = (np.count_nonzero(labelled_wells != detected_wells)) / (image.shape[0] * image.shape[1])
        self.assertTrue(mismatch_plate < 0.01)
        self.assertTrue(mismatch_wells < 0.05)

class TestExtractProfiles(unittest.TestCase):
    """
    Test function for identifying the first wells
    """

    def setUp(self):
        # First lets create a random 1D profile
        self.profile = np.random.randint(low=0, high=100, size=300)
        # Select indices where we want to identify peaks
        peak_positions = [2, 3, 70, 71, 72, 73, 74, 75, 98, 99, 100, 124, 125, 148, 149, 173, 174, 198, 199, 200, 222, 223, 224, 248, 249]
        for peak in peak_positions:
            # looping through we set the "mean" values here as higher
            self.profile[peak] = np.random.randint(low=2000, high=2500)
        # Repeat the above
        self.profile2 = np.random.randint(low=0, high=100, size=300)
        peak_positions = [2, 3, 98, 99, 100, 124, 125, 148, 149, 173, 174, 198, 199, 200, 222, 223, 224, 248, 249]
        # this peak will be detected but removed as thought to be too low
        low_peak = [70, 71, 72, 73]
        for peak in peak_positions:
            self.profile2[peak] = np.random.randint(low=2000, high=2500)
        for peak in low_peak:
            # we set this peak slightly lower so it gets filtered out before being re-added through the
            # interpolation check
            self.profile2[peak] = np.random.randint(low=600, high=1000)
        self.profile_diff = np.diff(self.profile)
        self.profile_diff2 = np.diff(self.profile2)
        self.first_peak = 75
        self.first_peak2 = 74
        self.median_gap = 25

    def test_find_first_well(self):
        ### TODO:  for this and other tests need examples that cover extremes and cover
        ### if statements, etc.
        peaks, gap = plate_detection.find_first_well(self.profile_diff, num_peaks = 8)
        self.assertEqual(peaks, self.first_peak)
        self.assertEqual(self.median_gap, gap)
        peaks, gap = plate_detection.find_first_well(self.profile_diff2, num_peaks = 8)
        self.assertEqual(peaks, self.first_peak2)
        self.assertEqual(self.median_gap, gap)

class TestMovePlateMask(unittest.TestCase):
    """
    Test function for moving the plate mask if it doesn't fit inside the well spacing
    """
    def setUp(self):
        self.img = generate_fake_plate_image(plate_length = 280, plate_ratio = 0.67)
        # Set coordinates of plates that have been "shifted"
        self.no_change = [22, 439, 75, 32, 312, 34]
        self.y_out1 = [22, 439, 75, 66, 344, 34]
        self.y_out2 = [22, 439, 75, 100, 378, 34]
        self.x_out1 = [97, 514, 75, 32, 312, 34]
        self.x_out2 = [172, 589, 75, 32, 312, 34]
        self.test_shifts = [self.no_change, self.y_out1, self.y_out2, self.x_out1, self.x_out2]

    def assign_values(self, list_of_args):
        """
        Simply takes a list and returns each value as correct variable
        """
        start_x = list_of_args[0]
        end_x = list_of_args[1]
        gap_x = list_of_args[2]
        start_y = list_of_args[3]
        end_y = list_of_args[4]
        gap_y = list_of_args[5]
        return start_x, end_x, gap_x, start_y, end_y, gap_y

    def test_move_plate_mask(self):
        for shift in self.test_shifts:
            start_x, end_x, gap_x, start_y, end_y, gap_y = self.assign_values(shift)
            start_x, end_x, start_y, end_y, = plate_detection.move_plate_mask(start_x, end_x, gap_x, start_y, end_y, gap_y, self.img)
            # Give ourselves 2 pixel error either direction
            self.assertTrue(20 <= start_x <= 24)
            self.assertTrue(437 <= end_x <= 441)
            self.assertTrue(30 <= start_y <= 34)
            self.assertTrue(310 <= end_y <= 314)

class TestGetFirstWellAndGaps(unittest.TestCase):
    """
    Test function for finding first the first well and the well spacing
    """

    def setUp(self):
        # The ratios here have been calculated based on the actual plate dimensions
        self.img50_hex = generate_fake_plate_image(plate_length = 200, plate_ratio = 0.436, hex=True, n_ver=10)
        self.img40_well = generate_fake_plate_image(plate_length = 280, plate_ratio = 0.67)
        self.img50_well = generate_fake_plate_image(plate_length = 280, plate_ratio = 0.96, n_ver = 10)

    def test_get_first_well_and_gaps_40_well(self):
        start_x, start_y, x_gap, y_gap = plate_detection.get_first_well_and_gaps(self.img40_well, 8, 5)
        self.assertTrue(62 <= start_x <= 64)
        self.assertTrue(38 <= start_y <= 40)
        self.assertTrue(75 <= x_gap <= 77)
        self.assertTrue(33 <= y_gap <= 35)


    def test_get_first_well_and_gaps_50_well(self):
        start_x, start_y, x_gap, y_gap = plate_detection.get_first_well_and_gaps(self.img50_well, 10, 5)
        self.assertTrue(112 <= start_x <= 116)
        self.assertTrue(38 <= start_y <= 40)
        self.assertTrue(52 <= x_gap <= 54)
        self.assertTrue(27 <= y_gap <= 29)

    def test_get_hex_plate_corners(self):
        start_x, end_x, start_y, end_y = plate_detection.get_corners_from_edges(self.img50_hex)
        self.assertTrue(16 <= start_x <= 20)
        self.assertTrue(58 <= start_y <= 60)
        self.assertTrue(441 <= end_x <= 445)
        self.assertTrue(282 <= end_y <= 286)
