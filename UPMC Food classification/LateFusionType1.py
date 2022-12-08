#!/usr/bin/env python
# coding: utf-8

# # Late Fusion Model Type 1 - Training BERT+LSTM and InceptionResNetV2+CNN together

# In[1]:


#command to activate auto complete
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
# Importing all libraies and external packages
import os
import re
import bert
import cv2
import sys
import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K 
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from sklearn.utils import shuffle
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.client import device_lib 
from keras.layers import  AveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model 
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
#checking if GPU is available
print(device_lib.list_local_devices())


# ## loading text dataset

# In[ ]:


#loading train and test data from csv files
text_features = ['image_path', 'text', 'food']
train_df = pd.read_csv('./texts/train_titles.csv', names=text_features, header=None, sep = ',', index_col=['image_path'])
test_df = pd.read_csv('./texts/test_titles.csv', names=text_features, header=None, sep = ',', index_col=['image_path'])
train_data.head()
test_data.head()
train_data = train_df.sort_values('image_path')
test_data = test_df.sort_values('image_path')
train_data.head()
test_data.head()

print("Number of training samples:",train_data.shape[0])
print("Number of test samples:",test_data.shape[0])


# In[ ]:


#Preprocessing Textual Data
def preprocess_text(sentence):
    #function to clean text data
    # Removing html tags
    sentence = remove_tags(sentence)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence
def remove_tags(text):
    return TAG_RE.sub('', text)
TAG_RE = re.compile(r'<[^>]+>')
preprocess_func = np.vectorize(preprocess_text)


# In[14]:


# Importing the pretrained BERT model
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


# ## loading Images dataset

# In[11]:


def get_missing(file, df):
  parts = file.split(os.sep)
  idx = parts[-1]
  cls = parts[-2]
  indexes = df[:,0]
  classes = df[:,2]

  if idx in indexes:
    text = df[idx == indexes][0,1]
    return pd.NA, pd.NA, pd.NA
  else:
    text = df[cls == classes][0,1]
    
  return idx, text, cls   
vec_get_missing = np.vectorize(get_missing, signature='(),(m,n)->(),(),()')  


# In[12]:


def add_not_found(path, df):
  files = glob.glob(path)
  df = df.reset_index()
  idxs, texts, cls = vec_get_missing(files, df.values)
  
  found = pd.DataFrame({"text": texts,
                        "food": cls,
                       "image_path": idxs})
  na = found.isna().sum().values[0]
  if na<found.shape[0]:
    df = df.append(found)
  df = df.drop_duplicates(subset='image_path', keep='first').dropna()
  df = df.set_index('image_path')
  df = shuffle(df, random_state = 0)
  return df      


# In[13]:


train = add_not_found('./images/train/*/*.jpg', train)
test = add_not_found('./images/test/*/*.jpg', test)

print("Number of training images:",train.shape[0])
print("Number of test images:",test.shape[0])


# In[15]:


def get_tokens(text, tokenizer):
  tokens = tokenizer.tokenize(text)
  tokens = ["[CLS]"] + tokens + ["[SEP]"]
  length = len(tokens)
  if length > max_length:
      tokens = tokens[:max_length]
  return tokens, length  
#generating masks for BERT input
def get_masks(text, tokenizer, max_length):
    tokens, length = get_tokens(text, tokenizer)
    return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))
vec_get_masks = np.vectorize(get_masks, signature = '(),(),()->(n)')
#generating segment ids for processing input acc to BERT reqs
def get_segments(text, tokenizer, max_length):
    tokens, length = get_tokens(text, tokenizer)
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.asarray(segments + [0] * (max_length - len(tokens)))
vec_get_segments = np.vectorize(get_segments, signature = '(),(),()->(n)')
#converting input text to ids accoridng to BERT vocabulary using BERT tokenizer
def get_ids(text, tokenizer, max_length):
    tokens, length = get_tokens(text, tokenizer)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = np.asarray(token_ids + [0] * (max_length-length))
    return input_ids
vec_get_ids = np.vectorize(get_ids, signature = '(),(),()->(n)')
#cretaing data function for BERT input
def get_texts(path):
    path = path.decode('utf-8')
    parts = path.split(os.sep)
    image_name = parts[-1]
    is_train = parts[-3] == 'train'
    if is_train:
      df = train
    else:
      df = test

    text = df['text'][image_name]
    return text
vec_get_text = np.vectorize(get_texts)

def prepare_text(paths):
    #Preparing texts
    
    texts = vec_get_text(paths)
    
    text_array = vec_preprocess_text(texts)
    ids = vec_get_ids(text_array, 
                      tokenizer, 
                      max_length).squeeze().astype(np.int32)
    masks = vec_get_masks(text_array,
                          tokenizer,
                          max_length).squeeze().astype(np.int32)
    segments = vec_get_segments(text_array,
                                tokenizer,
                                max_length).squeeze().astype(np.int32)
    
    return ids, segments, masks

def clean(i, tokens):
  try:
    this_token = tokens[i]
    next_token = tokens[i+1]
  except:
    return tokens
  if '##' in next_token:
      tokens.remove(next_token)
      tokens[i] = this_token + next_token[2:]
      tokens = clean(i, tokens)
      return tokens
  else:
    i = i+1
    tokens = clean(i, tokens)
    return tokens

def clean_text(array):
  array = array[(array!=0) & (array != 101) & (array != 102)]
  tokens = tokenizer.convert_ids_to_tokens(array)
  tokens = clean(0, tokens)
  text = ' '.join(tokens)
  return text


# In[16]:


# Images preprocessing
def load_image(path):
    path = path.decode('utf-8')
    image = cv2.imread(path)
    image = cv2.resize(image, (img_width, img_height))
    image = image/255
    image = image.astype(np.float32)
    parts = path.split(os.sep)
    labels = parts[-2] == Classes 
    labels = labels.astype(np.int32)
    return image, labels
vec_load_image = np.vectorize(load_image, signature = '()->(r,c,d),(s)')


# In[17]:


# Dataset creation

def prepare_data(paths):
    images, labels = tf.numpy_function(vec_load_image, [paths], [tf.float32, tf.int32])
    [ids, segments, masks, ] = tf.numpy_function(prepare_text, [paths], [tf.int32, tf.int32, tf.int32])
    images.set_shape([None, img_width, img_height, depth])
    labels.set_shape([None, nClasses])
    ids.set_shape([None, max_length])
    masks.set_shape([None, max_length])
    segments.set_shape([None, max_length])
    return ({"input_word_ids": ids, "input_mask": masks,  "segment_ids": segments, "image": images}, {"class": labels})
    return dataset


# In[18]:

batch_size =  64
#setting img size according to inception resnet input erequirements
img_width = 299
img_height = 299
depth = 3
max_length = 40
nClasses = train.food.nunique()
Classes = train.food.unique()
input_shape = (img_width, img_height, depth)


# In[19]:

#funstion to load images 
def tf_data(path, batch_size):
    paths = tf.data.Dataset.list_files(path)
    paths = paths.batch(64)
    dataset = paths.map(prepare_data, tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset   
data_train = tf_data('./images/train/*/*.jpg', batch_size)
data_test = tf_data('./images/test/*/*.jpg', batch_size)


# In[29]:

#Model Architecture

# Inception Resnet V2 model along with a CNN layer on top for image encoding
model_cnn = models.Sequential()
model_cnn.add(InceptionV3(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(299, 299, 3))))
model_cnn.add(layers.AveragePooling2D(pool_size=(1, 1), name='AVG_Pooling'))
model_cnn.add(layers.Dropout(.2, name='Dropout_0.2'))
model_cnn.add(layers.Conv2D(filters = 64, kernel_size = (1,1) , activation='relu'))
model_cnn.add(layers.AveragePooling2D(pool_size=(4, 4), name='AVG_Pooling2'))
model_cnn.add(layers.Dropout(.2, name='Dropout2_0.2'))
model_cnn.add(layers.Flatten(name='Flatten'))
#getting only activations here and not having a dimension equal to num of classes to train further
model_cnn.add(layers.Dense(256, name='Dense_256'))


# In[30]:


# Un freezing all layers to train teh entire model
for layer in model_cnn.layers:
    layer.trainable = True


# In[ ]:


# BERT + RNN layer on top for Txt encoding
input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
input_masks = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_masks")
input_segments = layers.Input(shape=(max_length,), dtype=tf.int32, name="segment_ids")
den_out, seq_out = bert_layer([input_ids, input_masks, input_segments])
X = layers.LSTM(128, name='LSTM')(seq_out)
X = layers.Dropout(0.5)(X)
X = layers.Dense(256, activation="relu")(X)
X = layers.Dropout(0.5)(X)
pred = layers.Dense(256, activation="relu")(X)
model_lstm = models.Model([input_ids, input_masks, input_segments], pred)

# In[ ]:

for layer in model_lstm.layers:
    layer.trainable = True

# In[36]:


#combining the two model sto train together

input_word_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = layers.Input(shape=(max_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                    name="segment_ids")
image_input = layers.Input(shape = input_shape, dtype=tf.float32,
                           name = "image")

image_side = model_cnn(image_input)
text_side = model_lstm([input_word_ids, input_mask, segment_ids])
# Concatenate features from images and texts and agging an ann on top
merged = layers.Concatenate()([image_side, text_side])
merged = layers.Dense(256, activation = 'relu')(merged)
merged = layers.Dropout(0.5)(merged)
merged = layers.Dense(128, activation = 'relu')(merged)
output = layers.Dense(nClasses, activation='softmax', name = "class")(merged)


# In[37]:


model = models.Model([input_word_ids, input_mask, segment_ids, image_input], output)


# In[39]:


model.summary()


# In[42]:


sgd = optimizers.SGD(learning_rate=0.0001)

#writing code to generate evaluation metrics
def recall_m(y_true, y_pred):
    #calculate recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
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

#compiling model
model.compile(loss = 'categorical_crossentropy', optimizer =sgd, 
              metrics = ['acc',f1_m,precision_m, recall_m])



# In[55]:


#Model training
history = model.fit(data_train, epochs=40, steps_per_epoch = train.shape[0]//batch_size,
                   validation_data = data_test, validation_steps = test.shape[0]//batch_size,)


# In[51]:


# Model evaluation 
model.evaluate(data_test, steps = test.shape[0]//batch_size)



