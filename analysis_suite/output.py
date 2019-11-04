"""
output.py file for analysis_suite

contains functions which export/save data
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def save_img(folder, filename, img, labelled_plate):
    plt.figure()
    plt.imshow(img)
    for well_lab in range(1, labelled_plate.max() + 1):
        col = plt.cm.gist_rainbow((well_lab / 9.1) % 1)
        plt.contour(labelled_plate == well_lab, levels=[0.5], colors=[col])

        bw0 = labelled_plate == well_lab
        pos0 = bw0.nonzero()
        pos = (np.min(pos0[0]), np.max(pos0[1]))
        plt.text(pos[1], pos[0], str(well_lab), color=col)
    plt.savefig(os.path.join(folder, filename))

def save_dict(folder, filename, dictionary):
    data = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Area', 'Mean Fluo', 'Total fluo'])
    data.to_csv(os.path.join(folder, "%s.csv"%(filename)))
