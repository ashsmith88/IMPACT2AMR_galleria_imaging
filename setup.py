import os
PWD = os.path.abspath(os.path.dirname(__file__))
print("CHANGING PATH")
print(PWD)
os.chdir(PWD)
MODNAME = "analysis_suite"

import matplotlib
matplotlib.use('qt5agg')
from setuptools import setup, Command, find_packages
#from distutils.core import setup

# Coverage command
import coverage
# For subprocesses (lazy) version
#import subprocess
import unittest



class CoverageCommand(Command):
    """Coverage Command"""
    description = "Run coverage on tests"
    user_options = []
    def initialize_options(self):
        """init options"""
        pass
    def finalize_options(self):
        """finalize options"""
        pass
    def run(self):
        """runner"""
        # TODO: Use API as follows for better control
        omitfiles = [
            os.path.join(MODNAME, "tests", "*"),
            os.path.join(MODNAME, "__main__.py"),
        ]
        for r, dirs, files in os.walk(MODNAME):
            omitfiles.extend(
                os.path.join(r, f) for f in
                filter(lambda f: f == "__init__.py", files)
            )

        cov = coverage.Coverage(
            source=[MODNAME],
            omit=omitfiles,
            )
        cov.start()
        # Run normal tests
        loader = unittest.TestLoader()
        tests = loader.discover(MODNAME)
        runner = unittest.runner.TextTestRunner()
        runner.run(tests)
        cov.save()
        cov.html_report()
        # Lazy way: call coverage!

setup(
    name='analysis_suite',
    version='0.1',
    description='Galleria Imaging module',
    author='Ashley Smith',
    author_email='ashleyasmith1991@gmail.com',
    #url='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    test_suite="analysis_suite.tests",
    cmdclass={
        "coverage":CoverageCommand,
        },
    )

#    setup_requires=[
        #"flake8",
#    ],
#    entry_points={
#        'console_scripts': ['mmhelper = mmhelper.gui:run_gui']
#    }
#)
