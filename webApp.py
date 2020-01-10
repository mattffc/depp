from flask import Flask, render_template      

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
'''
@app.route("/")
def home():
    return render_template("test.html")
    
@app.route("/salvador")
def salvador():
    return "Hello, Salvador"
    
#background process happening without any refreshing
@app.route('/background_process_test')
def background_process_test():
    print("Hello")
    return("nothing")
    
if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=80)
'''

'''
reuirements

numpy
pillow
Flask
urllib
os
werkzeug
'''

#depp learning kaggle copy
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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
#import matplotlib.pyplot as plt


#trainTrain = train[:1000].reset_index()
#trainValid = 

'''
train_generator=datagen.flow_from_dataframe(dataframe=train[:1000].reset_index(),directory=train_dir,x_col='id',
                                            y_col='label',class_mode='binary',batch_size=batch_size, target_size = (150,150))

print(train[1000:].head())
print(train[1000:].shape)
validation_generator=datagen.flow_from_dataframe(dataframe=train[1000:].reset_index(),directory=train_dir,x_col='id',
                                                y_col='label',class_mode='binary',batch_size=50, target_size = (150,150))
                                                
print("Cambs Team")
#Ok doing model down here
from keras import layers,models
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

#now lets do loss, optimzer and metrics
print("BSIB OH BSIB")
from keras import optimizers
model.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop(),metrics=['acc'])
print("here")
#train
epochs=1
history=model.fit_generator(train_generator,steps_per_epoch=1,epochs=epochs,validation_data=validation_generator,validation_steps=5)

#Lets plot the training
print("PLOT")

acc=history.history['acc']  ##getting  accuracy of each epochs
epochs_=range(0,epochs)    
plt.plot(epochs_,acc,label='training accuracy')
plt.xlabel('no of epochs')
plt.ylabel('accuracy')
plt.show()

acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs
plt.scatter(epochs_,acc_val,label="validation accuracy")
plt.title("no of epochs vs accuracy")
plt.legend()
plt.show()
'''
'''
#######
from keras.applications.vgg16 import VGG16
# vgg 16 is a really big NN
model_vg=VGG16(weights='imagenet',include_top=False)
#pretrained on imagenet database of images
model_vg.summary()

print("This is also new. BSIB")
'''
def extract_features_single(image):
    #print([1,image].shape)
    image = image*1.0/255.0
    from keras.applications.vgg16 import VGG16
    model_vg=VGG16(weights='imagenet',include_top=False)
    features = model_vg.predict(image)# [batch, width, height, rgb]
    print(features)
    
    return features
'''
def extract_features(directory,samples,df):
    
    
    features=np.zeros(shape=(samples,4,4,512))
    labels=np.zeros(shape=(samples))
    print("crash here")
    print(df.head())
    print(directory)
    generator=datagen.flow_from_dataframe(dataframe=df,directory=directory,x_col='id',
                                            y_col='label',class_mode='other',batch_size=batch_size,
                                            target_size=(150,150),shuffle=False)
    i=0
    print("len(generator)",len(generator))
    for input_batch,label_batch in generator:
        print(i)
        print("here666",input_batch.shape)
        feature_batch=model_vg.predict(input_batch)
        print("44",feature_batch.shape)
        print(features[i*batch_size:(i+1)*batch_size].shape)
        print(features.shape)
        features[i*batch_size:(i+1)*batch_size]=feature_batch
        labels[i*batch_size:(i+1)*batch_size]=label_batch
        i+=1
        if(i*batch_size>=samples):
            break
    return(features,labels)
'''
''''''

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
        K.clear_session()
        # load json and create model
        json_file = open('model100.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close() 
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model100.h5")
        print("Loaded model from disk")

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
            known_image = face_recognition.load_image_file("jon1.jpg")
            unknown_image = face_recognition.load_image_file("notJon.jpg")

            biden_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

            results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
            print(results)
            print("dlib prediction",results,str(predY[0][0])+results[0])
            return render_template("upload.html", user_image = full_filename, testText = str(predY[0][0])+results[0])
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
