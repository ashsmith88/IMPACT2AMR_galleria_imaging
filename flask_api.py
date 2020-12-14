import os
import shutil
from flask import Flask, request, abort, jsonify, send_from_directory, send_file
import json
import time

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

    data = main.run_batch(request_dir, "rect50")

    return timestamp

@app.route("/download/<path:path>")
def download_files(path):
    """Download a file."""
    return send_file(path, as_attachment=True)

@app.route("/getfiles/<path:path>")
def get_files(path):
    """Download a file."""
    path = os.path.join(path, "results")
    results_folder = os.path.join(os.getcwd(), path)
    all_files = [os.path.join(path, file) for file in os.listdir(results_folder)]
    return json.dumps(all_files)

@app.route("/cleanup/<path:path>")
def delete_data(path):
    shutil.rmtree(path)
    return

def create_directory():
    timestamp = str(time.time())
    timestamp = timestamp.split(".")[0]
    dir = os.path.join(upload_directory, timestamp)
    os.mkdir(dir)
    return dir, timestamp

if __name__ == '__main__':
    app.run(host='localhost', port=8080)
