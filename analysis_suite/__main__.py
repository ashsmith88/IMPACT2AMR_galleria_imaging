"""
The __main__.py file for galeria imaging

Takes the command line arguments and passes them into the main module
"""
# FILE        : __main__.py
# CREATED     : 30/09/2019
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>
#

import argparse
from analysis_suite.main import run_analysis, run_batch

PARSER = argparse.ArgumentParser(
    description="Image analysis suite for galleria images")
PARSER.add_argument("filename", type=str,
                    help="Data file(s) to load and analyse")
PARSER.add_argument("-plate", choices = ["rect50", "rect40", "hex50"],
                    help="The type of plate being used", default="rect50")
PARSER.add_argument("-batch", action="store_true",
                    help="A folder containing multiple image areas to run at the same time"
                    )
PARSER.add_argument("-exposure", "-e", type=str, default="300",
                    help="The exposure time used for bioluminescence, used to identify the files based on naming"
                    )
ARGS = PARSER.parse_args()

if ARGS.batch:
    run_batch(
        ARGS.filename,
        ARGS.plate,
        exposure=ARGS.exposure,
        )
else:
    run_analysis(
        ARGS.filename,
        ARGS.plate,
        )
