from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename  
import numpy as np 
import pandas as pd 
from PIL import Image
import os
import urllib.request
import face_recognition

UPLOAD_FOLDER = 'static'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# Check that the uploaded file has the right extension
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
    print("here")
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    # If a file is posted enter here
    if request.method == 'POST':
        # If no file found, return
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
            
            # Get the face encoding of the unknown image
            unknown_image = face_recognition.load_image_file(full_filename)
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
            
            known_images = ["depp1.jpg","depp2.jpg"]
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
            return render_template("upload.html", user_image = full_filename, testText = str(is_face_same))
            
    else:
        flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
    return redirect(request.url)

if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=80, debug=True,threaded=False)
