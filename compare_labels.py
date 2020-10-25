import sys
from skimage.external import tifffile
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import skimage.filters as skfilt
import scipy.ndimage as ndi
import numpy as np
import pandas as pd
from shutil import copyfile


def main(image_folder, label_folder):
    image_files = get_image_files(image_folder)
    #if os.path.isfile("Bad_image_labels.csv"):
    try:
        bad_image_labels = list(pd.read_csv("Bad_image_labels.csv")["Files"])
    except:
        bad_image_labels = []
    try:
        good_image_labels = list(pd.read_csv("Good_label_files.csv")["Files"])
    except:
        good_image_labels = []
    for i, file in enumerate(image_files, start=1):
        if (file in good_image_labels):
            continue
        img = load_file(image_folder, file)
        lbl = load_file(label_folder, file)
        #lbl = ndi.label(lbl)[0]
        plt.figure()
        plt.imshow(img)
        #plt.figure()
        #plt.imshow(lbl)
        col = plt.cm.gist_rainbow((2 / 9.1) % 1)
        # plot contour
        plt.contour(lbl == 2, levels=[0.5], colors=[col])
        plt.title("%s  image %s/%s"%(file, i,len(image_files)))

        callback = Index()
        imwrong = plt.axes([0.7, 0.05, 0.1, 0.075])
        imgood = plt.axes([0.81, 0.05, 0.1, 0.075])
        close = plt.axes([0.59, 0.05, 0.1, 0.075])
        bgood = Button(imgood, 'OK')
        bbad = Button(imwrong, 'Wrong')
        stopbutton = Button(close, "Stop")
        bgood.on_clicked(callback.image_good)
        bbad.on_clicked(callback.label_wrong)
        stopbutton.on_clicked(callback.stop_plots)
        plt.show()
        if callback.stop == True:
            update_incorrect_images_folder("wrong_images_folder", bad_image_labels, image_folder)
            return
        if callback.image_ok == False:
            if file in bad_image_labels:
                continue
            else:
                bad_image_labels.append(file)
        else:
            if file in bad_image_labels:
                bad_image_labels.remove(file)
            good_image_labels.append(file)

        # convert to pandas dataframe and save
        all_wrong_labels = pd.DataFrame(bad_image_labels, columns=["Files"])
        all_wrong_labels.to_csv("Bad_image_labels.csv", index=False)
        all_good_labels = pd.DataFrame(good_image_labels, columns=["Files"])
        all_good_labels.to_csv("Good_label_files.csv", index=False)
        update_incorrect_images_folder("wrong_images_folder", bad_image_labels, image_folder)

class Index(object):
    image_ok = True
    stop = False

    def image_good(self, event):
        plt.close()

    def label_wrong(self, event):
        self.image_ok = False
        plt.close()

    def stop_plots(self, event):
        self.stop=True
        plt.close()

def update_incorrect_images_folder(wrong_folder, wrong_files, well_folder):
    folder = os.path.join(os.path.dirname(os.path.dirname(well_folder)), wrong_folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    files_of_interest = [file for file in os.listdir(folder) if file.endswith(".tif")]
    different_files = [i for i in wrong_files + files_of_interest if i not in wrong_files or i not in files_of_interest]
    for file in different_files:
        if file in files_of_interest:
            os.remove(os.path.join(folder, file))
        else:
            img = os.path.join(well_folder, file)
            out = os.path.join(folder, file)
            copyfile(img, out)



def get_image_files(folder):
    files_of_interest = [file for file in os.listdir(folder) if file.endswith(".tif")]
    return files_of_interest

def load_file(folder, file):
    return np.array(tifffile.imread(os.path.join(folder, file)))


if __name__ == '__main__':
    image_folder = sys.argv[1]
    label_folder = sys.argv[2]
    main(image_folder, label_folder)
