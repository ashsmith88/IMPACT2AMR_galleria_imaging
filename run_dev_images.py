"""
Runs analysis on development images
"""

import os
from analysis_suite.main import run_analysis
import analysis_suite.loading as load

dev_folder = "dev_images"
files = [
    os.path.join(dev_folder, filename)
    for filename in
    ("bf_image.tif", "fluo_image.tif")
]
out_folder = load.create_out_folder("dev_images")
run_analysis(files, "rect50", out_folder)
