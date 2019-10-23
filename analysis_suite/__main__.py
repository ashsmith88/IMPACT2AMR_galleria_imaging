"""
The __main__.py file for galeria imaging

Takes the command line arguments and passes them into the main module
"""
# FILE        : __main__.py
# CREATED     : 30/09/2019
# AUTHOR      : A. Smith  <A.Smith@biosystemstechnology.com>
#

import argparse
from analysis_suite.main import run_analysis

PARSER = argparse.ArgumentParser(
    description="Image analysis suite for galleria images")
PARSER.add_argument("filename", nargs="+",
                    help="Data file(s) to load and analyse")
ARGS = PARSER.parse_args()


run_analysis(
        ARGS.filename,
        )
