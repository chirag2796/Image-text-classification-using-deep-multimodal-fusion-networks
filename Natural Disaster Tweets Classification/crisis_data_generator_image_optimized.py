# Taken from CrisisMMD preprocessor image generator from github

import numpy as np
import keras
from tensorflow.keras.utils import img_to_array
import warnings
import datetime
import optparse
import os, errno
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).

    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_file_list, text_vec, image_vec_dict, labels, max_seq_length=20, batch_size=32,
                 n_classes=2, shuffle=False):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.image_file_list = image_file_list
        self.text_vec = text_vec
        self.image_vec_dict = image_vec_dict
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.image_file_list) / float(self.batch_size)))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print(" index starts at: "+str(index * self.batch_size) +" ends at: "+str((index + 1) * self.batch_size))
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        # if(end > self.image_file_list):
        #     #print(" index starts at: " + str(start) + " ends at: " + str(end))
        #     end = self.image_file_list

        if(end > len(self.image_file_list)):
            #print(" index starts at: " + str(start) + " ends at: " + str(end))
            end = len(self.image_file_list)

        temp_indexes = self.indexes[start:end]
        #print(temp_indexes)
        # Generate data
        images_batch, text_batch, y = self.__data_generation(temp_indexes)

        return [images_batch, text_batch], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_file_list))
        #print(" indexes len: "+str(len(self.indexes)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        y = np.empty((len(indexes),self.n_classes), dtype=int)
        text_batch = np.empty((len(indexes), self.max_seq_length), dtype=int)
        images_batch = np.empty([len(indexes), 224, 224, 3])

        # Generate data
        for i, index in enumerate(indexes):
            #print(index)
            try:
                if(index <= len(self.image_file_list)):
                    # if (index in self.image_file_list):
                    image_file_name = str(self.image_file_list[index])
                    #print(image_file_name)
                    if(image_file_name in self.image_vec_dict):
                        # if(image_file_name=="image_null"):
                        #     img = np.zeros([1, 224, 224, 3])
                        # else:
                        img = self.image_vec_dict[image_file_name]
                        #img = image.load_img(self.image_file_list[index], target_size=(224, 224))
                        #img = image.img_to_array(img)
                        #img = np.expand_dims(img, axis=0)
                        #img = preprocess_input(img)
                        images_batch[i, :, :, :] = img
                        # Store class
                        y[i] = self.labels[index]
                        text_batch[i] = self.text_vec[index]
                    else:
                        #images_batch = np.delete(images_batch,i,0)
                        print("Exception assigining empty "+image_file_name)
                        img = np.zeros([1, 224, 224, 3])
                        images_batch[i, :, :, :] = img
                        #y = np.delete(y,i,0)
                        #text_batch=np.delete(text_batch,i,0)
                else:
                    print("Exception in indexing in image list less "+str(index)+" " +str(self.image_file_list) )
            except Exception as e:
                print("Exception in data generation.")
                print(e)
        current_images_batch = preprocess_input_vgg(images_batch)

        return current_images_batch, text_batch, y