# Import Library
from flask import Flask, Response, render_template, request, redirect, flash, send_file, send_from_directory
from PIL import Image
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model
import io
import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify, Response
import torch
import urllib.request
import requests

print(os.getcwd())
model = torch.hub.load('---/yolov5', 'custom', path='best.pt')

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})
    
    # if request.method == "POST":
    
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    p = model(img, size=1024)
    
    pred_img = p
    img_name = img.filename
    pred_img.render()

    for image in p.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(image)
        img_base64.save(bytes_io, format="jpeg")
        #imgs = [img]

    return Response(bytes_io.getvalue(), mimetype="image/jpeg")
    # return jsonify({"success" : "test"})
    
    
    # p = model(img)
    # p = cv2.cvtColor(p.imgs[0], cv2.COLOR_BGR2RGB)
    # return p

#     image_bytes = file.read()
#     img = Image.open(io.BytesIO(image_bytes))
#     pred_img = model(img)
#     img_name = img.filename

# for img in pred_img.ims:
#     bytes_io = io.BytesIO()
#     img_base64 = Image.fromarray(img)
#     img_base64.save(bytes_io, format="jpeg")
#     imgs = [img]

# return Response(bytes_io.getvalue(), mimetype="image/jpeg")     


if __name__ == "__main__":
    app.run(debug=True, port=5000)
