import os
import pickle

import numpy as np
import keras
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import concatenate
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import random
import aidrtokenize as aidrtokenize
from gensim.models import KeyedVectors
from keras.layers import Dense, Input, Dropout, Activation
from keras.models import Model
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint

class LoadDataset:
    @staticmethod
    def file_exist(file_name):
        if os.path.exists(file_name):
            return True
        else:
            return False

    @staticmethod
    def load_stopwords(file_name):
        stop_words = []
        with open(file_name, 'rU') as f:
            for line in f:
                line = line.strip()
                if (line == ""):
                    continue
                stop_words.append(line)
        return stop_words


    stop_words_file = "files/stop_words_english.txt"
    stop_words = load_stopwords(stop_words_file)

    @staticmethod
    def load_data_train(dataFile, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, delim):
        data = []
        labels = []
        with open(dataFile, 'rb') as f:
            next(f)
            for line in f:
                line = line.decode(encoding='utf-8', errors='strict')
                line = line.strip()
                if (line == ""):
                    continue
                row = line.split(delim)
                txt = row[3].strip().lower()
                txt = aidrtokenize.tokenize(txt)
                label = row[6]
                if (len(txt) < 1):
                    print (txt)
                    continue
                data.append(txt)
                labels.append(label)

        data_shuf = []
        lab_shuf = []
        index_shuf = list(range(len(data)))
        random.shuffle(index_shuf)
        for i in index_shuf:
            data_shuf.append(data[i])
            lab_shuf.append(labels[i])

        le = preprocessing.LabelEncoder()
        yL = le.fit_transform(lab_shuf)
        labels = list(le.classes_)

        label = yL.tolist()
        yC = len(set(label))
        yR = len(label)
        y = np.zeros((yR, yC))
        y[np.arange(yR), yL] = 1
        y = np.array(y, dtype=np.int32)

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV_TOK")
        tokenizer.fit_on_texts(data_shuf)
        sequences = tokenizer.texts_to_sequences(data_shuf)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        print('Shape of data tensor:', data.shape)
        return data, y, le, labels, word_index, tokenizer

    @staticmethod
    def load_data_val(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim, train_le):
        id_list=[]
        data = []
        labels = []
        with open(dataFile, 'rb') as f:
            next(f)
            for line in f:
                line = line.decode(encoding='utf-8', errors='strict')
                line = line.strip()
                if (line == ""):
                    continue
                row = line.split(delim)
                t_id= row[2].strip().lower()
                txt = row[3].strip().lower()
                txt = aidrtokenize.tokenize(txt)
                label = row[6]
                if (len(txt) < 1):
                    print (txt)
                    continue
                data.append(txt)
                labels.append(label)
                id_list.append(t_id)

        print(len(data))
        data_shuf = []
        lab_shuf = []
        index_shuf = list(range(len(data)))
        random.shuffle(index_shuf)
        for i in index_shuf:
            data_shuf.append(data[i])
            lab_shuf.append(labels[i])

        le = train_le  
        yL = le.transform(labels)
        labels = list(le.classes_)

        label = yL.tolist()
        yC = len(set(label))
        yR = len(label)
        y = np.zeros((yR, yC))
        y[np.arange(yR), yL] = 1
        y = np.array(y, dtype=np.int32)

        sequences = tokenizer.texts_to_sequences(data)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', data.shape)
        return data, y, le, labels, word_index,id_list


    @staticmethod
    def process_embeddings(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):
        nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM), dtype=np.float32)
        print(len(embedding_matrix))
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            try:
                if(word in model):
                    embedding_vector = model[word][0:EMBEDDING_DIM]
                    embedding_matrix[i] = np.asarray(embedding_vector, dtype=np.float32)
                else:
                    rng = np.random.RandomState()
                    embedding_vector = rng.randn(EMBEDDING_DIM)
                    embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
            except KeyError:
                try:
                    rng = np.random.RandomState()
                    embedding_vector = rng.randn(EMBEDDING_DIM) 
                    embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
                except KeyError:
                    continue
        return embedding_matrix


def is_file(filename):
    if os.path.exists(filename):
        return True
    else:
        return False

def model_save(model, model_dir, model_file_name, tokenizer, label_encoder):
    base_name = os.path.basename(model_file_name)
    base_name = os.path.splitext(base_name)[0]
    
    model_file = model_dir+"\\"+base_name +"_"+ ".hdf5"
    tokenizer_file = model_dir+"\\"+base_name +"_"+ ".tokenizer"
    label_encoder_file = model_dir+"\\"+base_name +"_"+ ".label_encoder"

    configfile = model_dir+"\\"+base_name+".config"
    configFile = open(configfile, "w")

    configFile.write("model_file="+model_file+"\n")
    configFile.write("tokenizer_file="+tokenizer_file+"\n")
    configFile.write("le_file="+label_encoder_file+"\n")
    configFile.close()

    files = []
    files.append(configfile)
    model.save(model_file)
    files.append(model_file)

    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(tokenizer_file)

    with open(label_encoder_file, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(label_encoder_file)

class CustomCNN:
    # Custom CNN taken from the sentence cnn by Y.kim
    
    @staticmethod    
    def custom_CNN(embedding_matrix,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,sequence_input):
        print('Preparing embedding matrix.')
        nb_words = min(MAX_NB_WORDS, len(word_index)+1)
        embedding_layer = Embedding(output_dim=EMBEDDING_DIM, input_dim=nb_words, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=True)
        embedded_sequences = embedding_layer(sequence_input)
        embedded_sequences = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)
        x = Conv2D(300, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
        x = MaxPool2D((MAX_SEQUENCE_LENGTH - 5 + 1, 1))(x)

        y = Conv2D(300, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
        y = MaxPool2D((MAX_SEQUENCE_LENGTH - 4 + 1, 1))(y)

        z = Conv2D(300, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
        z = MaxPool2D((MAX_SEQUENCE_LENGTH - 3 + 1, 1))(z)

        z1 = Conv2D(300, (2, EMBEDDING_DIM), activation='relu')(embedded_sequences)
        z1 = MaxPool2D((MAX_SEQUENCE_LENGTH - 2 + 1, 1))(z1)

        w1 = Conv2D(300, (1, EMBEDDING_DIM), activation='relu')(embedded_sequences)
        w1 = MaxPool2D((MAX_SEQUENCE_LENGTH - 1 + 1, 1))(w1)
        alpha = concatenate([w1,z1,z,y])

        merged_model = Flatten()(alpha)
        return merged_model

np.random.seed(42)


results_file = "output_file"
out_file = open(results_file, "w")

data_train_path = "data\\images_text_task1_train.tsv"
data_val_path = "data\\images_text_task1_val.tsv"
data_test_path = "data\\images_text_task1_test.tsv"

out_label_file_name = "label_file"


MAX_SEQUENCE_LENGTH = 25

x_train, y_train, train_le, train_labels, word_index, tokenizer = LoadDataset.load_data_train(data_train_path, 20000, MAX_SEQUENCE_LENGTH, "\t")
x_val, y_val, _, _, _,id_list = LoadDataset.load_data_val(data_val_path, tokenizer, MAX_SEQUENCE_LENGTH, "\t", train_le)
x_test, y_test, _, _, _,id_list = LoadDataset.load_data_val(data_test_path, tokenizer, MAX_SEQUENCE_LENGTH, "\t", train_le)

y_true = np.argmax(y_train, axis=1)
y_true = train_le.inverse_transform(y_true)

nb_classes = len(set(y_true.tolist()))

print ("Labels number: "+str(nb_classes))


# Word2Vec embeddings
model_file = "E:\Dev\Multimodal\multimodal_social_media\models\crisisNLP_word2vec_model\crisisNLP_word_vector.bin"
emb_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
embedding_matrix = LoadDataset.process_embeddings(word_index, emb_model, 20000, 300)
print("Embedding shape: "+str(embedding_matrix.shape))

inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
cnn = CustomCNN.custom_CNN(embedding_matrix, word_index, 20000, 300, MAX_SEQUENCE_LENGTH, inputs)




network = Activation('relu')(cnn)
network = Dropout(0.05)(network)
network = Dense(128)(network)
network = Activation('relu')(network)
network = Dense(64)(network)
network = Activation('relu')(network)

out = Dense(nb_classes, activation='softmax',name='lrec-softmax')(network)
model = Model(inputs=inputs, outputs=out)

lr = 0.00005
adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)


model.load_weights("E:\\Dev\\Multimodal\\multimodal_social_media\\models\\task_informative_text_img_agreed_lab_train_text\\informativeness_cnn_keras_13-11-2022_05-02-39.hdf5")

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='max')
checkpoint = ModelCheckpoint("models", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [early_stopping, checkpoint]

print(model.summary())

history = model.fit([x_train], y_train, epochs=10, batch_size=8, validation_data=([x_val], y_val), callbacks=callbacks_list, verbose=1)



print ("Model loaded successfully")

dir_name = os.path.dirname("models")
base_name = os.path.basename(data_train_path)
base_name = os.path.splitext(base_name)[0]
model_dir = dir_name+"\\"+base_name+"_text"

result_val = model.predict([x_val], batch_size=8, verbose=1)
result_test = model.predict([x_test], batch_size=8, verbose=1)

model_save(model, model_dir, "models", tokenizer, train_le)