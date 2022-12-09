import os
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import concatenate
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.applications.vgg16 import preprocess_input
import metrics as metrics
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from gensim.models import KeyedVectors
from keras.layers import Input, Dropout
import data_processor_fusion as data_process_fusion
import pickle
from tensorflow.keras.layers import BatchNormalization
from data_generator_fusion import DataGenerator
import keras


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



def model_save(model, model_dir, model_file_name, tokenizer, label_encoder):
    base_name = os.path.basename(model_file_name)
    base_name = os.path.splitext(base_name)[0]
    model_file = model_dir + "/" + base_name + ".hdf5"
    tokenizer_file = model_dir + "/" + base_name + ".tokenizer"
    label_encoder_file = model_dir + "/" + base_name + ".label_encoder"

    configfile = model_dir + "/" + base_name + ".config"
    configFile = open(configfile, "w")
    configFile.write("model_file=" + model_file + "\n")
    configFile.write("tokenizer_file=" + tokenizer_file + "\n")
    configFile.write("label_encoder_file=" + label_encoder_file + "\n")
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


data_train_path = "data\\images_text_task1_train.tsv"
data_val_path = "data\\images_text_task1_val.tsv"
data_test_path = "data\\images_text_task1_test.tsv"

MAX_SEQUENCE_LENGTH = 25
vocab_size = 20000
embed_dim = 300
batch_size = 8
n_epoch = 14

dir_name = os.path.dirname(data_train_path)
base_name = os.path.basename(data_train_path)
base_name = os.path.splitext(base_name)[0]

image_mappings_filepath = "files\\images_path_mapping.txt"
image_list = []
with open(image_mappings_filepath, 'rU') as f:
    for line in f:
        line = line.strip()
        if (line == ""):
            continue
        row = line.split("\t")
        image_path = row[0].strip()
        image_list.append(image_path)
images_npy_data = {} 
for i ,img_path in enumerate(image_list):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_data = tf.keras.utils.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    images_npy_data[img_path] = img_data

train_x, train_image_list, train_y, train_le, train_labels, word_index, tokenizer = data_process_fusion.read_train_data_multimodal(
    data_train_path,
    vocab_size,
    MAX_SEQUENCE_LENGTH, 6,
    "\t")


base_name = os.path.basename(data_val_path)
base_name = os.path.splitext(base_name)[0]


dev_x, dev_image_list, val_y, dev_le, dev_labels, _ = data_process_fusion.read_dev_data_multimodal(data_val_path, tokenizer, MAX_SEQUENCE_LENGTH, 6, "\t")

n_classes = len(set(train_labels))
print ("Number of labels: " + str(n_classes))
params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
            "n_classes": n_classes, "shuffle": True}
train_data_generator = DataGenerator(train_image_list, train_x, images_npy_data, train_y, **params)

params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
            "n_classes": n_classes, "shuffle": False}
val_data_generator = DataGenerator(dev_image_list, dev_x, images_npy_data, val_y, **params)



# Word2vec embeddings
model_file = "E:\Dev\Multimodal\multimodal_social_media\models\crisisNLP_word2vec_model\crisisNLP_word_vector.bin"
emb_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
embedding_matrix = data_process_fusion.prepare_embedding(word_index, emb_model, vocab_size,
                                                    embed_dim)

inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))

cnn = CustomCNN.custom_CNN(embedding_matrix, word_index, vocab_size, embed_dim,
                        MAX_SEQUENCE_LENGTH, inputs)

text_network = Dense(1000, activation='relu')(cnn)
text_network = BatchNormalization()(text_network)

# Saving the last layer outputs in pickle files to be used in the lateral linear model

last_layer_output_text_data_file = open('fusion_2_last_layer_output_text.h5', 'wb')
pickle.dump(text_network, last_layer_output_text_data_file)
last_layer_output_text_data_file.close()


vgg16 = VGG16(weights='imagenet')
last_layer_output = vgg16.get_layer('fc2').output


last_layer_output = Dense(1000, activation='relu')(last_layer_output)
last_layer_output = BatchNormalization()(last_layer_output)

last_layer_output_image_data_file = open('fusion_2_last_layer_output_image.h5', 'wb')
pickle.dump(last_layer_output, last_layer_output_image_data_file)
last_layer_output_image_data_file.close()

# exit()

# Combining of image and text network nodels
combined_model = concatenate([last_layer_output, text_network], axis=-1)
combined_model = BatchNormalization()(combined_model)
combined_model = Dropout(0.4)(combined_model)
combined_model = Dense(512, activation='relu')(combined_model)
combined_model = Dropout(0.25)(combined_model)
combined_model = Dense(100, activation='relu')(combined_model)
combined_model = Dropout(0.025)(combined_model)
out = Dense(n_classes, activation='softmax')(combined_model)
model = Model(inputs=[vgg16.input, inputs], outputs=out)


lr = 0.00002
adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print(model.summary())

early_stopping_callback = callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
lr_reduce = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1,
                                            factor=0.1, min_lr=0.0001,mode='max')


checkpoint = ModelCheckpoint("models", monitor='val_acc', verbose=1,
                                save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [early_stopping_callback, lr_reduce, checkpoint]
history = model.fit_generator(generator=train_data_generator, epochs=n_epoch, validation_data=val_data_generator,
                                use_multiprocessing=True,
                                workers=2, verbose=1, callbacks=callbacks_list)


dir_name = os.path.dirname("models")
base_name = os.path.basename(data_train_path)
base_name = os.path.splitext(base_name)[0]
model_dir = dir_name + "/" + base_name
model_save(model, model_dir, "model_fusion", tokenizer, train_le)

# Testing on test dataset
results_val = model.predict_generator(val_data_generator, verbose=1)
AUC, accuracy, P, R, F1, results_detail = metrics.calculate_metrics(val_y, results_val, train_le)

result = str("{0:.4f}".format(accuracy)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
    "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"
print(result)
print (results_detail)


test_x, test_image_list, test_y, test_le, test_labels, ids = data_process_fusion.read_dev_data_multimodal(data_test_path, tokenizer, MAX_SEQUENCE_LENGTH, 6, "\t")

params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
            "n_classes": n_classes, "shuffle": False}

test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)

test_prob = model.predict_generator(test_data_generator, verbose=1)

AUC, accuracy, P, R, F1, results_detail = metrics.calculate_metrics(val_y, results_val, train_le)
result = str("{0:.4f}".format(accuracy)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
    "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"
print(result)
print (results_detail)


AUC, accuracy, P, R, F1, results_detail = metrics.calculate_metrics(test_y, test_prob, train_le)
result = str("{0:.4f}".format(accuracy)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
    "{0:.2f}".format(R)) + "\t" + str("{0:.4f}".format(F1)) + "\t" + str("{0:.4f}".format(AUC))+ "\n"
print("results of model:\t"+base_name+"\t"+result)
print (results_detail)
