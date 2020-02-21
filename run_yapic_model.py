from yapic.session import Session
from yapic_io.prediction_batch import PredictionBatch
import os
import sys
import time
import numpy as np
import tensorflow as tf
from analysis_suite.loading import load_tiff_file
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def run_model(model, in_folder, out_folder):
    session = Session()
    session.load_prediction_data(in_folder, out_folder)
    session.load_model(model)
    session.set_normalization('local')


    #run_batch(session, in_folder)
    predict(session)

def run_batch(session, in_folder):
    images = []
    files = get_file_list(in_folder)
    for f in files:
        img = load_tiff_file(f)
        images.append(np.expand_dims(img, axis=0))
    input = np.vstack(images)
    #input = np.array(images)
    print("Predicting", flush=True)
    start = time.time()
    preds = session.model.predict(input)
    end = time.time()
    print("Done in %s seconds"%(end-start))

def predict(session):
    data_predict = PredictionBatch(session.dataset,
                                   2,
                                   session.output_tile_size_zxy,
                                   session.padding_zxy)
    data_predict.set_normalize_mode('local')
    data_predict.set_pixel_dimension_order('bzxyc')


    data_predict = np.vstack([data.pixels() for data in data_predict])
    print("Predicting", flush=True)
    start = time.time()
    preds = session.model.predict(data_predict)
    end = time.time()
    print("Done in %s seconds"%(end-start))

    """
    for n, item in enumerate(data_predict, start=1):
        print("Processing image %s of %s"%(n, len(data_predict)))
        start = time.time()
        result = session.model.predict(item.pixels())
        pred = time.time()
        print("Predicted in %s seconds"%(pred-start))
        item.put_probmap_data(result)
        end = time.time()
        print("Done in %s seconds"%(end-start))
    """

def get_file_list(in_folder):
    files = [os.path.join(in_folder, file) for file in os.listdir(in_folder) if file.endswith('.tif')]
    return files

if __name__ == '__main__':
    model = sys.argv[1]
    in_folder = sys.argv[2]
    out_folder = sys.argv[3]
    run_model(model, in_folder, out_folder)
