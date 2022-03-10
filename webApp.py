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

# Check that the uploaded file has the right extension
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def get_depp_encodings():
    img_filepaths = [DEPP_FOLDER/f for f in os.listdir(DEPP_FOLDER) if allowed_file(f)]
    imgs = [face_recognition.load_image_file(fp) for fp in img_filepaths]
    return [face_recognition.face_encodings(img)[0] for img in imgs]

depp_encodings = get_depp_encodings()

def is_image_depp(filepath):
    unknown_image = face_recognition.load_image_file(filepath)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    # If len==0 => no people, therefore no Depp :(
    if len(unknown_encodings) == 0:
        return False
    # Check unkown encoding against all known Depp encodings
    for depp_encoding in depp_encodings:
        is_depp = face_recognition.compare_faces([depp_encoding], unknown_encodings[0])[0]
        if is_depp:
            return True
    return False

@app.route('/')
def upload_form():
    print("here")
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    # If a file is posted enter here
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If file empty name, return
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        # If file correctly uploaded, enter here
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save the file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # See if it is Depp
            is_depp = is_image_depp(full_filename)
            
            # Get the face encoding of the unknown image
            unknown_image = face_recognition.load_image_file(full_filename)
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
            
            known_images = ["depp1.jpeg","depp2.jpg"]
            is_face_same = False
            for known_image_name in known_images:
                # Get the face image of the known depp image
                known_image = face_recognition.load_image_file(known_image_name)
                
                # Get the encodings
                known_encoding = face_recognition.face_encodings(known_image)[0]
                
                # Compare the encodings of the known and unknown faces
                is_face_same = face_recognition.compare_faces([known_encoding], unknown_encoding)[0]
                print(is_face_same)
                # If there is a match for any of the known faces, break
                if is_face_same:
                    break

            # Return the template with the results
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
    print("hello")
    app.run(host='0.0.0.0', port=80, debug=True,threaded=False)
