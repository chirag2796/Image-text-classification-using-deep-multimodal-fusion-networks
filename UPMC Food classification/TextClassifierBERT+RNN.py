#command to activate auto complete
%config Completer.use_jedi = False
# Importing all libraies and external packages
import os
import re
import bert
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.client import device_lib 
#checking if GPU is available
print(device_lib.list_local_devices())

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

#Encoding Textual data to categorical variables and preprocessing it
encoder = LabelEncoder()
train_preprocessed = preprocess_func(train_data.text.values)
test_preprocessed = preprocess_func(test_data.text.values)

encoded_labels_train = encoder.fit_transform(train_data.food.values)
labels_train = utils.to_categorical(encoded_labels_train, train_data.food.nunique())

encoded_labels_test = encoder.fit_transform(test_data.food.values)
labels_test = utils.to_categorical(encoded_labels_test, train_data.food.nunique())

print("Processed text sample:", train_preprocessed[0])
print("Shape of train labels:", labels_train.shape)

# Importing the pretrained BERT model
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
#generating masks for BERT input
def get_masks(text, max_length):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]

    return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))
vec_get_masks = np.vectorize(get_masks, signature = '(),()->(n)')
#generating segment ids for processing input acc to BERT reqs
def get_segments(text, max_length):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]
    
    segments = []
    current_segment_id = 0
    with_tags = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.asarray(segments + [0] * (max_length - len(tokens)))
vec_get_segments = np.vectorize(get_segments, signature = '(),()->(n)')
#converting input text to ids accoridng to BERT vocabulary using BERT tokenizer
def get_ids(text, tokenizer, max_length):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = np.asarray(token_ids + [0] * (max_length-length))
    return input_ids
vec_get_ids = np.vectorize(get_ids, signature = '(),(),()->(n)')
#cretaing data function for BERT input
def prepare(text_array, tokenizer, max_length = 128):
    
    ids = vec_get_ids(text_array, 
                      tokenizer, 
                      max_length).squeeze()
    masks = vec_get_masks(text_array,
                      max_length).squeeze()
    segments = vec_get_segments(text_array,
                      max_length).squeeze()

    return ids, segments, masks

#Generating Input for the BERT Model
max_length = 40 
ids_train, segments_train, masks_train = prepare(train_preprocessed, tokenizer, max_length)
ids_test, segments_test, masks_test = prepare(train_preprocessed, tokenizer, max_length)
#creating input layer for ids, segment ids and masked tokens
input_word_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
input_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_masks")
segment_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name="segment_ids")
#passing inputs through BERT pretrained model and generating output
den_out, seq_out = bert_layer([input_word_ids, input_mask, segment_ids])

#Model architecture RNN
X = layers.LSTM(128)(seq_out) #appending LSTM after BERT pretrained
X = layers.Dropout(0.5)(X)
X = layers.Dense(256, activation="relu")(X)
X = layers.Dropout(0.5)(X)
output = layers.Dense(train_data.food.nunique(), activation = 'softmax')(X)
model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[output])

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

#Model training parameters 
model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr=.001), 
              metrics = ['acc',f1_m,precision_m, recall_m])
#training model
history = model.fit([ids_train, masks_train, segments_train], 
          labels_train,
          epochs = 20,
          batch_size = 512,
          validation_split = 0.2)

#Evaluating model
model.evaluate([ids_test, masks_test, segments_test],labels_test, batch_size = 512)


