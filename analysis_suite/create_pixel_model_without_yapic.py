import sys
import os

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow import keras
from skimage.external import tifffile

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def create_model(in_folder, labelled_folder, epochs=2):
    """
    Creates a keras model
    Parameters
    ------
    in_files : list
        A list of files containing raw data
    labelled_files : list
        A list of files containing corresponding label images
    epochs : int, opt
        Number of epochs to run (default : 2)
    Returns
    ------
    mod : keras model
        The generated model
    """

    in_files = [os.path.join(in_folder, file) for file in os.listdir(in_folder) if file.endswith(".tif")]
    labelled_files = [os.path.join(labelled_folder, file) for file in os.listdir(labelled_folder) if file.endswith(".tif")]

    all_data = []
    all_labels = []
    # loop through the input file and label files
    for n, (in_file, label_file) in enumerate(zip(sorted(in_files), sorted(labelled_files))):
        if n == 100:
            continue
        # get the correct data
        data_arr, label_arr = get_data(in_file, label_file)

        # scale it and reshape it
        #data_arr = data_arr / 255
        #data_arr = data_arr.reshape((data_arr.shape[0], 1, data_arr.shape[1]))

        # append the current data to all data
        all_data.append(data_arr)
        all_labels.append(label_arr)

    # if only a single input file then we simply use this!
    if len(all_data) == 1:
        all_data = data_arr
        all_labels = label_arr
    else: # otherwise we need to combine all the data in to a single array and the same with the labels
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

    for i in range(1, 2):
        print("label :%s has %s entries"%(i, len(all_labels[all_labels==i])))

    # Define the parameters of the model
    mod = keras.Sequential([
        #keras.layers.Flatten(input_shape=(1, 1)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='sigmoid')])

    # Define the accuracy metrics and parameters
    mod.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Run the model
    #model.fit(xTrain, yTrain, epochs=2)#, batch_size=4)
    mod.fit(all_data, all_labels, epochs=epochs, batch_size=1000)
    mod.save("blah.h5")
    return mod

def predict_data(model_file, data_file_to_predict):
    """
    Makes classification predictions on new data based on a pre-exisiting model
    Parameters
    ------
    mod : keras model
        The keras model we will use to predict
    data_file_to_predict : path
        path to the data file which we will
    """
    mod = keras.models.load_model(model_file)
    predict_data = tifffile.imread(data_file_to_predict)
    predictImg = predict_data.flatten()

    predicted = mod.predict(predictImg)
    #predicted = np.argmax(model.predict(x), axis=-1)

    #get the labels
    predicted = np.argmax(predicted, axis=-1)

    # reshape to the correct size array
    prediction = np.reshape(predicted, (predict_data.shape[0], predict_data.shape[1]))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(predict_data)
    plt.figure()
    plt.imshow(prediction)
    plt.show()

    return prediction


def get_data(data_file, labelled_file):
    """
    Gets array data from tifffiles
    Parameters
    ------
    data_file : path
        path to the file containing the raw data
    label_file : path
        path to the file containing the label data
    Returns
    ------
    featuresImg : ndarray
        array containing the raw data
    labelImg : ndarray
        array containing a corresponding labelled image
    nBands : int
        Number of bands in the image
    """
    data = tifffile.imread(data_file)

    labelled = tifffile.imread(labelled_file)

    #print(data.shape)
    #print(labelled.shape)

    #featuresImg= changeDimension(data)
    #labelImg = changeDimension(labelled)

    return data.flatten(), labelled.flatten()

if __name__ == '__main__':
    in_folder = sys.argv[1]
    out_folder = sys.argv[2]
    #predict_data(in_folder, out_folder)
    create_model(in_folder, out_folder)
