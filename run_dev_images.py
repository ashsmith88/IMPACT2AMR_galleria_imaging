"""
Runs analysis on development images
"""

from analysis_suite.main import run_analysis
import analysis_suite.loading as load

files = ["dev_images\\bf_image.tif", "dev_images\\fluo_image.tif"]
out_folder = load.create_out_folder("dev_images")
run_analysis(files, "rect50", out_folder)
