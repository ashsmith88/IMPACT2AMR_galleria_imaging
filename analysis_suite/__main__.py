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
PARSER.add_argument("filename", nargs="+",
                    help="Data file(s) to load and analyse")
PARSER.add_argument("-batch", action="store_true",
                    help="a folder containing multiple image areas to run at the same time"
                    )
ARGS = PARSER.parse_args()


if ARGS.batch:
    run_batch(
        ARGS.filename,
        )
else:
    run_analysis(
        ARGS.filename,
        )
