import os
import shutil
from flask import Flask, request, abort, jsonify, send_from_directory, send_file
import json
import time
from waitress import serve

import sys

sys.path.append(".")

import analysis_suite.main as main

app = Flask(__name__)

upload_directory = os.getcwd()
if not os.path.exists(upload_directory):
    os.mkdir(upload_directory)

@app.route('/upload',methods=['GET','POST'])
def upload():
    request_dir, timestamp = create_directory()
    #request_dir = upload_directory
    for f in request.files:
        filename = request.files[f]
        filename.save(os.path.join(request_dir, filename.filename))

    dataframes = main.run_batch(request_dir, "rect50")

    all_data = {}
    for meas, df in dataframes.items():
        all_data[meas] = df.to_dict("index")
    all_data['folder'] = timestamp

    return json.dumps(all_data)

@app.route("/download/<path:path>")
def download_files(path):
    """Download a file."""
    files_to_return = get_files(path)
    new_dir = os.path.join(path, "images_to_return")
    os.mkdir(new_dir)
    for file in files_to_return:
        os.rename(file, os.path.join(new_dir, "%s"%(os.path.basename(file))))
    shutil.make_archive(os.path.join(path, "zipped_images"), 'zip', new_dir)
    file = os.path.join(path, "zipped_images.zip")
    return send_file(file, as_attachment=True)

def get_files(path):
    path = os.path.join(path, "results")
    results_folder = os.path.join(os.getcwd(), path)
    all_files = [os.path.join(path, file) for file in os.listdir(results_folder) if file.endswith(".jpg")]
    return all_files

@app.route("/cleanup/<path:path>")
def delete_data(path):
    shutil.rmtree(path)
    return ""

def create_directory():
    timestamp = str(time.time())
    timestamp = timestamp.split(".")[0]
    dir = os.path.join(upload_directory, timestamp)
    os.mkdir(dir)
    return dir, timestamp

if __name__ == '__main__':
    #app.run(host='localhost', port=8080) # for development

    serve(app, host='0.0.0.0',port=8080) # for deployment
    #serve(app, host='0.0.0.0', port=5000) # for deployment with specific port
