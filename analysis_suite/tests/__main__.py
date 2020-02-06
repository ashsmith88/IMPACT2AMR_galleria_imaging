"""
Initiates the unittests
"""
# By default run standard test discovery, but add a call to
# matplotlib.pyplot.show
import sys
from unittest import main
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sys.path.append("..")

    # unittest.TestLoader
    main(module=None, exit=False, verbosity=2)
    plt.show()
