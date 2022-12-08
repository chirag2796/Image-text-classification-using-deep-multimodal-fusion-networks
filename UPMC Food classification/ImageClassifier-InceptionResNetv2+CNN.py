#command to activate auto complete
%config Completer.use_jedi = False
# Importing all libraies and external packages
import os
import cv2
import sys
import time
import pickle
import numpy as np
import pandas as pd
from keras import backend as K 
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from keras.layers import  AveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model 
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2

#checking if GPU is available
print(device_lib.list_local_devices())

train_dir = 'images/train'
val_dir = 'images/test'
num_train_samples = len(os.listdir(train_dir)) #67988 
num_val_samples = len(os.listdir(val_dir)) #22716
num_classes = 101
batch_size = 64

# keeping image format with respect to keras functions requirement
img_width = 400
img_height = 400
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
    

#Defining Train data generator and adding augmentation for training: horizontal, vertical shift, horizontal, vertical flips and brightness shift to normalize all images with different brightness
train_datagen = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3, horizontal_flip=True,  
                vertical_flip=True, brightness_range=(0.2, 0.8))
#Defining Test data generator, augmentations not needed during Testing
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width),                                   batch_size=batch_size, class_mode='categorical')

val_generator = test_datagen.flow_from_directory(val_dir, target_size=(img_height, img_width),
                batch_size=batch_size, class_mode='categorical')

# Model Architecture
#InceptionResNet layer using pretrained Imagenet weights
#include_top removes the last fully connected layer from the pretrained model
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
#CNN architecture on top of pretrained model
x = base_model.output
x = AveragePooling2D(pool_size=(1, 1))(x)
x = Dropout(.2)(x)
x = Conv2D(filters = 64, kernel_size = (1,1) , activation='relu')(x)
x = AveragePooling2D(pool_size=(4, 4))(x)
x = Dropout(.2)(x)
x = Flatten()(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(base_model.input, output)

#writing code to generate evaluation metrics
def recall_m(y_true, y_pred):
    #calculate recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #recall tp/tp+
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    #calculate precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    #calculate f1 score
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Training the model 
model_image.compile( optimizer=SGD(lr=.01, momentum=.9), loss='categorical_crossentropy', 
    metrics = ['acc',f1_m,precision_m, recall_m])

model.fit(train_generator,  steps_per_epoch = num_train_samples // batch_size,
          val_data=val_generator,  val_steps=num_val_samples // batch_size,
          epochs= 80)

#Evaluating the model
model_image.evaluate_generator( val_generator, num_val_samples/batch_size)
