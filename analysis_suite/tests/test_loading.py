"""
Unittests for data_loading
"""

import unittest
import os
import shutil
import tempfile
from analysis_suite import data_loading as load

class TestOutputFileAndFolder(unittest.TestCase):
    """
    Test function for making the output file and folder
    """

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.file = os.path.join(self.output_dir, "T2 image 161091")
        self.out_file = "bioluminescence_reading_T2"


    def test_get_out_file(self):
        out_file = load.get_out_file(self.file)
        self.assertEqual(self.out_file, out_file)

    def test_create_out_folder_exists(self):
        self.results_folder = os.path.join(self.output_dir, "results")
        os.mkdir(self.results_folder)
        self.out_folder = load.create_out_folder(self.output_dir)
        self.assertEqual(self.results_folder, self.out_folder)
        os.rmdir(self.results_folder)
        if os.path.isdir(self.out_folder):
            os.rmdir(self.out_folder)

    def test_create_out_folder_doesnt_exist(self):
        self.results_folder = os.path.join(self.output_dir, "results")
        self.out_folder1 = load.create_out_folder(self.output_dir)
        self.assertEqual(self.results_folder, self.out_folder1)
        os.rmdir(self.results_folder)
        if os.path.isdir(self.out_folder1):
            os.rmdir(self.out_folder1)

    def tearDown(self):
        os.rmdir(self.output_dir)

class TestGetImageFiles(unittest.TestCase):
    """
    Test function for extracting image files and their respective time points from a folder
    """

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.folder2 = tempfile.mkdtemp()
        self.exposure_time = "300"
        file1 = "T0 image 161119"
        file2 = "T0 300 sec 161119"
        file3 = "T0 10 sec 161119"
        file4 = "T5 image 161119"
        file5 = "T5 300 sec 161119"
        file6 = "T5 10 sec 161119"
        self.files = [file1, file2, file3, file4, file5, file6]
        self.outfile_300 = [[os.path.join(self.folder, (file1 + '.1sc')), os.path.join(self.folder, (file2 + '.1sc'))],
                                [os.path.join(self.folder, (file4 + '.1sc')), os.path.join(self.folder, (file5 + '.1sc'))]]
        self.outfile_10 = [[os.path.join(self.folder2, (file1 + '.tif')), os.path.join(self.folder2, (file3 + '.tif'))],
                                [os.path.join(self.folder2, (file4 + '.tif')), os.path.join(self.folder2, (file6 + '.tif'))]]

        for f in self.files:
            with open(os.path.join(self.folder, (f + '.1sc')), 'w'):
                pass

        for f in self.files:
            with open(os.path.join(self.folder2, (f + '.tif')), 'w'):
                pass

        self.tpoints = [0, 5]

    def test_get_image_files_300(self):
        out_files, out_tpoints = load.get_image_files(self.folder)
        self.assertEqual(out_files, self.outfile_300)
        self.assertEqual(out_tpoints, self.tpoints)

    def test_get_image_files_10(self):
        out_files, out_tpoints = load.get_image_files(self.folder2, exposure_time='10', filetype=".tif")
        self.assertEqual(out_files, self.outfile_10)
        self.assertEqual(out_tpoints, self.tpoints)

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)
        shutil.rmtree(self.folder2, ignore_errors=True)
