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
import h5py

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt

train=pd.read_csv(r'C:\Users\mattf\OneDrive\Documents\deppLearning\deppLabels.csv')
train.label=train.label.astype(str)
train_dir=r'C:\Users\mattf\OneDrive\Documents\deppLearning\combine'
test_dir=r"C:\Users\mattf\OneDrive\Documents\deppLearning\combineTest"

labels = pd.read_csv(r'C:\Users\mattf\OneDrive\Documents\deppLearning\deppLabels.csv')

for i in range(0,min(1,len(labels))):
    train_imgs = os.listdir(r'C:\Users\mattf\OneDrive\Documents\deppLearning\combine')
    labelledim = labels.iloc[i]
    print(r'C:\Users\mattf\OneDrive\Documents\deppLearning\combine\\' + labelledim.id)
    im = Image.open(r'C:\Users\mattf\OneDrive\Documents\deppLearning\combine\\' + labelledim.id)

    #plt.imshow(im)
    #plt.title("Is Johnny: {:}".format("Yes" if labelledim.label else "No"))
    #plt.show()
    
print("BSIB BABY")
print("MATT TEST")
#get data generators up here


datagen=ImageDataGenerator(rescale=1./255)
batch_size=150

print(train.shape)
print(train.head())
train = train[:1500]
trainOrig = train
trainOrig.label=trainOrig.label.astype(int)
##train = train.sample(frac=1)
print(train.head())

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
#######
from keras.applications.vgg16 import VGG16
# vgg 16 is a really big NN
model_vg=VGG16(weights='imagenet',include_top=False)
#pretrained on imagenet database of images
model_vg.summary()

print("This is also new. BSIB")

def extract_features_single(image):
    #print([1,image].shape)
    image = image*1.0/255.0
    from keras.applications.vgg16 import VGG16
    model_vg=VGG16(weights='imagenet',include_top=False)
    print("input_batch[0]33",image[0,:,:,0])
    features = model_vg.predict(image)# [batch, width, height, rgb]
    print(features)
    
    return features

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
        print("input_batch[0]22",input_batch[0,:,:,0])
        feature_batch=model_vg.predict(input_batch)
        print("44",feature_batch[0].shape)
        print("this is the first from feature batch",feature_batch[0,:,:,0])
        
        print(features[i*batch_size:(i+1)*batch_size].shape)
        print(features.shape)
        features[i*batch_size:(i+1)*batch_size]=feature_batch
        labels[i*batch_size:(i+1)*batch_size]=label_batch
        i+=1
        if(i*batch_size>=samples):
            break
    return(features,labels)
    
def doTraining():
    # new pipeline, look for face with VGG and draw bounding box (don't have any face box training examples), if no face found - not johnny,
    # next take input crop of face train new VGG on these cropped faces
    train.label=train.label.astype(int)
    print("train",train[:10],train[-10:],train_dir)
    saving = False
    if saving:
        features,labels=extract_features(train_dir,1500,train)
        #save features and labels to disk
        print("shapes before",features.shape,labels.shape)
        h5f = h5py.File('dataFeats.h5', 'w')
        h5f.create_dataset('dataset_1', data=features)
        h5f.create_dataset('dataset_2', data=labels)
        h5f.close()
    
    h5f = h5py.File('dataFeats.h5','r')
    features = h5f['dataset_1'][:]
    labels = h5f['dataset_2'][:]
    #print(b[:5],c[:5])
    #print("shapes after",b.shape,c.shape)
    h5f.close()
    
    train_features=features[:1000]
    train_labels=labels[:1000]

    validation_features=features[998:]
    validation_labels=labels[998:]
    print("finished doing the feature extraction")
    print("333444",train_features.shape,validation_features.shape)

    #test_features,test_labels=extract_features(test_dir,4000,df_test)

    #reshape all the vgg topless outputs from 3d arrays to 1D
    train_features=train_features.reshape((-1,4*4*512))
    validation_features=validation_features.reshape((-1,4*4*512))

    #test_features=test_features.reshape((4000,4*4*512))
    # make a simple fully connected model with 1 layer 212 hidden units
    from keras import regularizers
    modelling = False
    if modelling:
        model = models.Sequential()
        model.add(layers.Dense(212,activation='relu',kernel_regularizer=regularizers.l1_l2(.001),input_dim=(4*4*512)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1,activation='sigmoid'))
        model.summary()

        model.compile(optimizer=optimizers.rmsprop(),loss='binary_crossentropy',metrics=['acc'])
        history=model.fit(train_features,train_labels,epochs=100,batch_size=15,validation_data=(validation_features,validation_labels))
        #######

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
    
    # load json and create model
    json_file = open('model100.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close() 
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model100.h5")
    print("Loaded model from disk")


    
    predFeatures,predLabels = extract_features(test_dir,300,trainOrig)#[:155])
    predFeatures=predFeatures.reshape((-1,4*4*512))
    predY = loaded_model.predict(predFeatures)
    
    predicted_class_indices=np.argmax(predY,axis=1)
    print("predY before round",predY[:5])
    predY = np.around(predY)
    print(predY[:5])

    plt.plot(predY)
    plt.show()
    from sklearn import metrics
    print(trainOrig[:151].shape)
    print(trainOrig[:5])
    print(trainOrig[:5].values)
    print(trainOrig.values[:5,1])
    print('Mean Absolute Error:', metrics.mean_absolute_error(trainOrig.values[:5,1], predY[:5]))
    print('Mean Absolute Error:', metrics.mean_absolute_error(predLabels, predY))
    
    #something about extract features single makes it give a different prediciton to what it did before
    image = Image.open(r"C:\Users\mattf\OneDrive\Documents\deppLearning\combineTest\img_0.jpeg")
    #image = image.resize((basewidth, hsize))
    image = np.array(image.resize((150,150)))
    image = np.expand_dims(image, axis=0)
    print("777",image.shape)
    
    predFeatures = extract_features_single(image)
    print(predFeatures.shape)
    print(predFeatures[0,:,:,0])
    l=lp
    #predFeatures,predLabels = extract_features(test_dir,1,trainOrig)
    predFeatures=predFeatures.reshape((-1,4*4*512))
    predY = loaded_model.predict(predFeatures)
    print(predY)
    return
doTraining()