from flask import Flask, render_template      

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras import layers,models
from keras import optimizers
from keras.models import model_from_json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

def extract_features_single(image):
    #print([1,image].shape)
    image = image*1.0/255.0
    from keras.applications.vgg16 import VGG16
    model_vg=VGG16(weights='imagenet',include_top=False)
    features = model_vg.predict(image)# [batch, width, height, rgb]
    print(features)
    
    return features

import os
#import magic
import urllib.request
#from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
#from deppLearningModel import extract_features_single
from keras.models import model_from_json
import face_recognition
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
from keras import backend as K






def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
    print("here")
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        #Before prediction
        #K.clear_session()
        # load json and create model
        #json_file = open('model100.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close() 
        #loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        #loaded_model.load_weights("model100.h5")
        #print("Loaded model from disk")

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            '''
            #print(type(file.read()))
            #print(np.array(Image.open(file)).shape)
            
            image = Image.open(file)
            #image = image.resize((basewidth, hsize))
            image = np.array(image.resize((150,150)))
            image = np.expand_dims(image, axis=0)
            print("777",image.shape)
            
            # resize image, do prediction on image
            
            predFeatures = extract_features_single(image)
            predFeatures=predFeatures.reshape((-1,4*4*512))
            predY = loaded_model.predict(predFeatures)
            
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(full_filename)
            #After prediction
            K.clear_session()
            print("predY",predY)
            '''
            known_image = face_recognition.load_image_file("depp1.jpg")
            unknown_image = face_recognition.load_image_file(full_filename)

            biden_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

            results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
            print(results)
            
            return render_template("upload.html", user_image = full_filename, testText = str(results[0]))
            #return redirect('/')
    else:
        flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
    return redirect(request.url)

if __name__ == "__main__":
    '''
    image = np.array(Image.open("img_0.jpeg").resize((150,150)))
    image = np.expand_dims(image, axis=0)
    predFeatures = extract_features_single(image)
    predFeatures=predFeatures.reshape((-1,4*4*512))
    predY = loaded_model.predict(predFeatures)
    print(predY)
    '''
    app.run(host='0.0.0.0', port=80, debug=True,threaded=False)
