"""
Contains a class that stores information on the plate dimensions
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

class Plate(object):
    """
    Stores information on the plate being used that can then be accessed and used to
    create masks, etc.

    For instance, detecting the plate outline will allow for the wells to be populated
    automatically
    """

    def __init__(
            self,
            plate_type = "rect",
            ):
        """
        Attributes
        ------
        plate_type : str
            Specifies which type of plate ("rect" or "hex")
        plate_dimensions : tuple
            2D Dimensions of the plate in mm (height (y), width (x))
        well_dimensions : tuple
            2D Dimensions of the wells in mm (height (y), width (x))
        first_well : tuple
            start points for the first well (top left) in mm
        col_space : float
            spacing between the start of each column of wells in mm
        row_space : float
            spacing between the start of each row of wells in mm
        """

        self.plate_type = plate_type

        self.set_plate_type()

        # Hidden attributes

    def set_plate_type(self):
        """
        Gets the correct plate information based on the well type selected
        """
        dimensions_dict = self._get_correct_dimensions()
        self.plate_dimensions = dimensions_dict["plate_dim"]
        self.well_dimensions = dimensions_dict["well_dim"]
        self._first_well = dimensions_dict["first_well"]
        self._col_space = dimensions_dict["col_space"]
        self._row_space = dimensions_dict["row_space"]
        self._row_space_no_stagger = dimensions_dict["row_space_no_stagger"]
        self._no_columns = dimensions_dict["ncols"]
        self._no_rows = dimensions_dict["nrows"]

    def _get_correct_dimensions(self):
        """
        Contains hard corded dimension information of the plates in dictionaries
        depending on the plate seleted the dictionary returned can be used to set class attributes
        """

        hex_plate_50 = {"plate_dim":(85, 195), "well_dim":(17.5, 15), "first_well":(5, 12.92),
            "col_space" : 17, "row_space" : 29, "row_space_no_stagger" : 14.75, "nrows" : 10, "ncols" : 5} ## TODO: need to measure row space no stagger
        rect_plate_40 = {"plate_dim":(85.65, 127), "well_dim":(8, 20), "first_well":(3.83, 10.5),
            "col_space" : 22.5, "row_space" : 10, "row_space_no_stagger" : 10, "nrows" : 8, "ncols" : 5}
        rect_plate_50 = {"plate_dim":(135, 140), "well_dim":(9, 21), "first_well":(5, 12.5),
            "col_space" : 25, "row_space" : 13, "row_space_no_stagger" : 13, "nrows" : 10, "ncols" : 5}

        if self.plate_type == "rect40":
            return rect_plate_40
        elif self.plate_type == "hex50":
            return hex_plate_50
        elif self.plate_type == "rect50":
            return rect_plate_50

    def get_plate_corners(self, first_well_x, first_well_y, x_gap, y_gap):
        """
        Gets the plate corners based on the location of the first well and spacings

        Parameters
        ------
        first_well_x : int
            x pixel location of the top left corner of first well
        first_well_y : int
            y pixel location of the top left corner of first well
        x_gap : int
            gap between wells along the x axis (in pixels)
        y_gap : int
            gap between wells along the y axis (in pixels)
        """
        # First get the ratio
        x_ratio = x_gap / self._col_space
        y_ratio = y_gap / self._row_space_no_stagger
        # get the top left corner based on the first well location and ratio
        start_x = first_well_x - (self._first_well[1] * x_ratio)
        start_y = first_well_y - (self._first_well[0] * y_ratio)
        # get the bottom left corner based on the top left corner, plate dimensions and ratio
        end_x = start_x + (self.plate_dimensions[1] * x_ratio)
        end_y = start_y + (self.plate_dimensions[0] * y_ratio)

        return int(start_x), int(end_x), int(start_y), int(end_y)


    def _calibrate_plate(self, plate_im=None):
        """
        Gets the calibration values between the plate in mm and pixels
        """
        # Get the dimensions of the plate
        self.plate_dim_pixels = tuple((plate_im.shape[0], plate_im.shape[1]))

        # Get the calibration value between pixels and mm
        self._y_cal = self.plate_dimensions[0] / self.plate_dim_pixels[0]
        self._x_cal = self.plate_dimensions[1] / self.plate_dim_pixels[1]

    def locate_wells(self, plate_im, plate_type=None):
        """
        Takes the detected plate image, determines the location of the wells
        and creates a mask with these wells

        #### TODO: Check this is what it actually returns

        Parameters
        ------
        plate_im : ndarray
            A 2D array representing the face of the the detected plate (# TODO: Check this is input)
        """
        # First set the plate values to 0 as currently 1
        plate_im = plate_im - 1

        self._calibrate_plate(plate_im=plate_im)
        if plate_type:
            self.set_plate_type(plate_type)

        # Get the location of the first well and its dimensions in pixels
        self._first_well_pixels = tuple((round(self._first_well[0] / self._y_cal), round(self._first_well[1] / self._x_cal)))
        self._well_dim_pixels = tuple((round(self.well_dimensions[0] / self._y_cal), round(self.well_dimensions[1] / self._x_cal)))

        # Get the location of all wells (i.e. the top left corner pixels)
        x_vals, y_vals = self._get_well_pixel_locations(plate_im)

        #Generate the correct mask
        if "rect" in self.plate_type:
            self.well_mask = np.zeros(self._well_dim_pixels) + 1
        elif "hex" in self.plate_type:
            self.well_mask = self._create_hexagon(base_shape = self._well_dim_pixels)
            # for the hexagonal plate every other row is staggered so we need to calculate
            # the location of these rows separately by adding half of the x and y spacing to the first well
            staggered_rows = tuple(np.array(self._first_well_pixels) +
                np.array(tuple(((self._row_space/ 2 / self._y_cal), (self._col_space / 2 / self._x_cal)))))
            x_vals_staggered, y_vals_staggered = self._get_well_pixel_locations(plate_im, first_well_pix = staggered_rows)
            # add well masks
            plate_im = self._add_well_masks(plate_im, x_vals_staggered, y_vals_staggered)

        plate_im = self._add_well_masks(plate_im, x_vals, y_vals)
        self.plate = plate_im + 1

    def _get_well_pixel_locations(self, plate_im, first_well_pix = None):
        """
        Gets the location of the wells (in pixels) depending on the first well location
        Uses the spacing thats been calculated and interpolates until the end of the plate in
        both x and y
        """

        # If the first well is provided then use this (allows for staggered rows in hexagon plates)
        if first_well_pix is None:
            first_well_pix = self._first_well_pixels

        # Calculate spacing in pixels
        y_space_pix = (self._row_space/ self._y_cal)
        x_space_pix = (self._col_space / self._x_cal)

        # Generate x and y initial coords as lists
        x_vals = [int(round(first_well_pix[1] + (x_space_pix * n))) for n in range(30) if
            (first_well_pix[1] + (x_space_pix * n)) < (self.plate_dim_pixels[1] - self._well_dim_pixels[1])]

        y_vals = [int(round(first_well_pix[0] + (y_space_pix * n))) for n in range(30) if
            (first_well_pix[0] + (y_space_pix * n)) < (self.plate_dim_pixels[0] - self._well_dim_pixels[0])]

        return x_vals, y_vals

    def _add_well_masks(self, plate_im, x_vals, y_vals):
        """
        Adds well masks to the plate background
        """
        # Loop through combinations of x and y coordinates for "top left" corner of wells and add well mask
        for x_coo in x_vals:
            for y_coo in y_vals:
                # We need to add the well to the existing background before replacing the information
                # This is for the hexagonal plates where the actual rectangular array outlines overlap
                # By adding them together it keeps the previous corners (rather than replacing them)
                well = np.add(plate_im[y_coo : (y_coo + self._well_dim_pixels[0]), x_coo : (x_coo + self._well_dim_pixels[1])], self.well_mask)
                # Now we can set the array to the correct plate location
                plate_im[y_coo : (y_coo + self._well_dim_pixels[0]), x_coo : (x_coo + self._well_dim_pixels[1])] = well

        return plate_im

    def _create_hexagon(self, base_shape = (17, 15)):
        """
        Creates a rectangular array and uses linear regression to "cut corners off"
        resulting in a hexagonal centre
        """
        # Create a square base for the hexagon and set all pixels to 1
        # Also need to reverse it as it gets rotated back to the correct shape at the end
        rect_base = np.zeros(tuple(reversed(base_shape))) + 1

        ### Determine optimum point to "cut" corner
        optimal_length = None
        best_difference = None
        # base of the hexagon is currently on the x axis so loop through different size of this
        for bottom_length in range(rect_base.shape[0]):
            # this determines the length of the edge from the corner to the start of the hexagon base
            triangle_edge = (rect_base.shape[1] - bottom_length) / 2
            # The triangle corners that need to be cut must go from the edge of the base to
            # the centre of the y - axis. Lets use pythagoras to calculate hypotenuse length
            hypotenuse_length = math.sqrt((triangle_edge ** 2) + ((rect_base.shape[0] / 2) ** 2))
            # calculate the difference between the base length and hypotenuse
            difference = abs(bottom_length - hypotenuse_length)
            # when the difference is lowest (closest in length) this is optimum base length
            if best_difference is None:
                best_difference = difference
            if difference < best_difference:
                optimal_length = bottom_length
                best_difference = difference

        # use the optimum length to re calculate the length of the corner to cut on x-axis
        corner_cut_length_x = (rect_base.shape[1] - optimal_length) / 2

        # Get the coordinates for each corner
        all_coords = []
        coords = self._get_corner_points(corner_cut_length_x, rect_base.shape, corner= "top_right", mirror=True)
        all_coords.append(coords)
        coords = self._get_corner_points(corner_cut_length_x, rect_base.shape, corner = "top_right")#, mirror=True)
        all_coords.append(coords)
        coords = self._get_corner_points(corner_cut_length_x, rect_base.shape, corner = "bottom_right", mirror=True)
        all_coords.append(coords)
        coords = self._get_corner_points(corner_cut_length_x, rect_base.shape, corner = "bottom_right")#, mirror=True)
        all_coords.append(coords)

        # Set each coordinate in the list to 0
        for coords in all_coords:
            for coo_ in coords:
                try:
                    rect_base[coo_] = 0
                except:
                    print(coo_)

        # return rotated hexagon
        return np.rot90(rect_base)


    def _get_corner_points(self, corner_cut_length, base_shape, corner = "top_right", mirror = False):
        """
        Determines the coordinates of the pixels in the corner between a point on the x-axis and
        a point on the y-axis

        Parameters
        ------
        corner_cut_length : int
            The length along the x-axis that needs to be cut off
        base_shape : tuple
            shape of the array
        corner : str (optional)
            specifies the corner, options "top_right" and "bottom_right"
        mirror : boolean
            If True then it does the opposite corner (i.e. top right = top left)
        """

        if corner == "top_right":
            # Get points at intersection with x and y axis
            x_points = np.array([(base_shape[1] - corner_cut_length), (base_shape[1])]).reshape(-1, 1)
            y_points = np.array([0, (base_shape[0] / 2)]).reshape(-1, 1)
        if corner == "bottom_right":
            x_points = np.array([(base_shape[1] - corner_cut_length), (base_shape[1]-0)]).reshape(-1, 1)
            y_points = np.array([base_shape[0]-1, (base_shape[0] / 2)]).reshape(-1, 1)

        # specify the start and end of all x_axis points
        start = int(base_shape[1] - corner_cut_length)
        end = int(base_shape[1])

        # perform linear regression between x and y intercepts to create straight "cut"
        lm = LinearRegression(fit_intercept=True)
        lm.fit(x_points, y_points)
        x_coords = range(start, end)

        coords = []
        # We can now loop through all the x coordinates and predict the max or min
        # (depending if top or bottom corner) y value for the cut
        for n, x_val in enumerate(x_coords, start=0):
            pred = int(lm.predict(np.array(x_val).reshape(-1, 1))[0][0])
            if mirror:
                coords.append((pred, (base_shape[1]-x_val)-1))
            else:
                coords.append((pred, x_val))
            # Get list of all y coordinates between top or bottom of array and
            # y value of interest
            if "top" in corner:
                y_vals = range(0, pred)
            else:
                y_vals = range(pred, base_shape[0])
            # Now we can append coordinates for all the y_vals between the max or min (depending
            # if top or bottom)
            for y in y_vals:
                if mirror:
                    coords.append((y, (base_shape[1]-x_val)-1))
                else:
                    coords.append((y, x_val))
        return coords
