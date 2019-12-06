"""
Passes files into the detection pipeline just for testing
"""
import os
import sys
sys.path.append('D:\\BiosystemsTechnology\\galleria_imaging')
from detection import detect_plate
import analysis_suite.BR_reader.reader as biorad_reader

images_dict = {"D:\\BiosystemsTechnology\\Images\\test_images_development\\hexagon_50_wells.1sc" : "hex50",
                "D:\\BiosystemsTechnology\\Images\\test_images_development\\plate_40_wells.1sc" : "rect40",
                    "D:\\BiosystemsTechnology\\Images\\test_images_development\\plate_40_wells_2.1sc" : "rect40"}

for filepath, plate in images_dict.items():
    if os.path.isfile(filepath):
        img = biorad_reader.Reader(filepath)
        loaded_image = img.get_image()
    detect_plate(loaded_image, plate_type=plate)
