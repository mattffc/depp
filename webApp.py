import os
from flask import Flask, render_template      
import numpy as np 
import pandas as pd 
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path
import face_recognition

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
UPLOAD_FOLDER = Path("static")
DEPP_FOLDER = Path("deppFaces")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def get_depp_encoding():
    depp_img_filepath = DEPP_FOLDER/"1.jpg"
    depp_image = face_recognition.load_image_file(depp_img_filepath)
    return [face_recognition.face_encodings(depp_image)[0]]

depp_encoding = get_depp_encoding()

def is_image_depp(filepath):
    unknown_image = face_recognition.load_image_file(filepath)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    if len(unknown_encodings)==0:
        return False
    results = face_recognition.compare_faces(depp_encoding, unknown_encodings[0])
    if len(results)==1 and results[0]==True:
        return True
    else:
        return False

@app.route('/')
def upload_form():
    print("here")
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = Path(secure_filename(file.filename))
            file.save(str(app.config['UPLOAD_FOLDER']/filename))
            flash('File successfully uploaded')

            full_filename = app.config['UPLOAD_FOLDER']/filename
            is_depp = is_image_depp(full_filename)

            return render_template(
                    "upload.html", 
                    user_image = full_filename, 
                    testText = getStrForResult(is_depp))
    else:
        flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
    return redirect(request.url)

def getStrForResult(is_depp):
    if is_depp:
        return "This is Depp"
    else:
        return "This is not Depp"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True,threaded=False)
