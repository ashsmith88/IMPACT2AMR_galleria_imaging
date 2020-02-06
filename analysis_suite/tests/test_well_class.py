"""
Tests for custom well class
"""
import unittest
import numpy as np
import pandas as pd
from analysis_suite.well_class import SingleWell, AllWells

class TestAllWellsClass(unittest.TestCase):
    """
    Test function for the AllWells Class
    """

    def setUp(self):
        # list of args for each test = in the order [well_num, tpoint, area, mean_fluo, total_fluo]
        self.input0 = [1, 0, 20, 50, 1000]
        self.input1 = [1, 1, 15, 5, 75]
        self.input2 = [2, 0, 10, 8, 80]

        self.area_dict  = {0 : 20, 1 : 15}
        self.mean_fluo_dict = {0 : 50, 1 : 5}
        self.total_fluo_dict = {0 : 1000, 1 : 75}

        self.area_dict2 = {0 : 10}
        self.mean_fluo_dict2 = {0 : 8}
        self.total_fluo_dict2 = {0 : 80}

        area = {1: [20, 15], 2: [10, np.nan]}
        self.area = pd.DataFrame.from_dict(data=area, orient='index', columns=[0, 1])

        fluo = {1: [50, 5], 2: [8, np.nan]}
        self.fluo = pd.DataFrame.from_dict(data=fluo, orient='index', columns=[0, 1])

        fluo_total = {1: [1000, 75], 2: [80, np.nan]}
        self.fluo_total = pd.DataFrame.from_dict(data=fluo_total, orient='index', columns=[0, 1])

        self.allwells = AllWells()
        self.allwells.add_well_info(self.input0[0], self.input0[1], self.input0[2], self.input0[3], self.input0[4])
        self.allwells.add_well_info(self.input1[0], self.input1[1], self.input1[2], self.input1[3], self.input1[4])
        self.allwells.add_well_info(self.input2[0], self.input2[1], self.input2[2], self.input2[3], self.input2[4])
        self.allwells.create_dataframes()


    def test_add_well_info(self):

        self.assertEqual(self.area_dict, self.allwells.wells[1].area_dict)
        self.assertEqual(self.mean_fluo_dict, self.allwells.wells[1].mean_fluo_dict)
        self.assertEqual(self.total_fluo_dict, self.allwells.wells[1].total_fluo_dict)
        self.assertEqual(self.area_dict2, self.allwells.wells[2].area_dict)
        self.assertEqual(self.mean_fluo_dict2, self.allwells.wells[2].mean_fluo_dict)
        self.assertEqual(self.total_fluo_dict2, self.allwells.wells[2].total_fluo_dict)

    def test_create_dataframes(self):
        pd.testing.assert_frame_equal(self.allwells.dataframes['area_dict'], self.area)
        pd.testing.assert_frame_equal(self.allwells.dataframes['mean_fluo_dict'], self.fluo)
        pd.testing.assert_frame_equal(self.allwells.dataframes['total_fluo_dict'], self.fluo_total)



class TestSingleWellClass(unittest.TestCase):
    """
    Test function for the SingleWell class
    """

    def setUp(self):
        # list of args for each test = in the order [tpoint, area, mean_fluo, total_fluo]
        self.input0 = [0, 20, 50, 1000]
        self.input1 = [1, 15, 5, 75]

        self.area_dict = {0 : 20, 1 : 15}
        self.mean_fluo_dict = {0 : 50, 1 : 5}
        self.total_fluo_dict = {0 : 1000, 1 : 75}

    def test_add_tpoint_data(self):
        well = SingleWell()
        well.add_tpoint_data(self.input0[0], self.input0[1], self.input0[2], self.input0[3])
        well.add_tpoint_data(self.input1[0], self.input1[1], self.input1[2], self.input1[3])
        self.assertEqual(self.area_dict, well.area_dict)
        self.assertEqual(self.mean_fluo_dict, well.mean_fluo_dict)
        self.assertEqual(self.total_fluo_dict, well.total_fluo_dict)
