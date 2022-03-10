import os
from flask import Flask, render_template      
import numpy as np 
import pandas as pd 
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path
import cv2

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

@app.route('/')
def upload_form():
    print("hi")
    return "yo"
    
@app.route("/test", methods=['POST'])
def test_method():         
    # print(request.json)
    print(dir(request))
    print(request.files)
    print("type",type(request.files))
    print(dir(request.files))
    print(request.files["media"])
             
    # get the base64 encoded string
    im_b64 = request.json['image']
@app.route("/test2", methods=['POST'])
def test_2():   
    r = request
    # convert string of image data to uint8
    #nparr = np.fromstring(r.data, np.uint8)
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img)
    print(img.shape)
    print(img[0,0,:])
    print("here")

if __name__ == "__main__":
    print("hello")
    app.run(host='0.0.0.0', port=80, debug=True,threaded=False)