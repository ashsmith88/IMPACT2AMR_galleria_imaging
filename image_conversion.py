import sys
from skimage.external import tifffile
import os
import matplotlib.pyplot as plt
import skimage.filters as skfilt
import scipy.ndimage as ndi

def main(image_folder, label_folder):
    image_files = get_image_files(image_folder)
    label_files = get_image_files(label_folder)
    convert_files(image_files)
    convert_label_files(label_files)

def convert_files(files):
    for file in files:
        img = load_file(file)
        if (img.shape[0] < 200) or (img.shape[1] < 200):
            img = ndi.zoom(img, 3)
            print("Converting: %s to size: %s"%(file, img.shape))
            tifffile.imsave(file, img)

def convert_label_files(files):
    for file in files:
        img = load_file(file)
        if img.ndim != 2:
            img = img[...,-1].astype('uint8')
            img = ndi.label(img)[0] + 1
            tifffile.imsave(file, img.astype("uint8"))
        if (img.shape[0] < 200) or (img.shape[1] < 200):
            img = ndi.zoom(img, 3)
            print("Converting: %s to size: %s"%(file, img.shape))
            tifffile.imsave(file, img)

def load_file(img_path):
    return tifffile.imread(img_path)

def get_image_files(folder):
    files_of_interest = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".tif")]
    return files_of_interest

if __name__ == '__main__':
    image_folder = sys.argv[1]
    label_folder = sys.argv[2]
    main(image_folder, label_folder)
