from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import metrics as metrics
from keras.utils import to_categorical
from sklearn import preprocessing
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf

        
def instantiate_vgg_model(n_classes):
    adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    
    vgg16 = VGG16(weights='imagenet')
    fc2 = vgg16.get_layer('fc2').output
    prediction = Dense(n_classes, activation='softmax', name='predictions')(fc2)
    model = Model(vgg16.input, prediction)

    print(model.summary())
    
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
            
    
def generate_data_file(data_file,delim="\t"):
    "Generateion code taken from the original dataset preprocessing code"
    data_list=[]
    label_list=[]
    id_list=[]
    with open(data_file, 'rU', encoding="utf8") as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            tweet_id=row[0]            
            img_path=row[4]
            label=row[7]
            data_list.append(img_path)
            label_list.append(label)
            id_list.append(tweet_id)
    image_len = len(data_list)
    all_images = np.empty([image_len, 224, 224, 3])
    all_labels = []
    for i in range(image_len):
           img = tf.keras.utils.load_img(data_list[i], target_size=(224, 224))
           img = tf.keras.utils.img_to_array(img)
           img = np.expand_dims(img, axis=0)
           img = preprocess_input(img)        
           lab = label_list[i]        
           all_images[i, :, :, :] = img
           all_labels.append (lab)
    n_classes=len(set(all_labels))   
    print("Number of labels: "+str(n_classes))    
    le = preprocessing.LabelEncoder()
    y=le.fit_transform(all_labels) 
    y=np.asarray(y)
    all_labels = to_categorical(y, n_classeses=n_classes)
    print(all_labels.shape)    
    return all_images,all_labels,id_list,le,n_classes



batch_size = 8
n_epoch=12

data_train_path = "data\\images_text_task1_train.tsv"
data_val_path = "data\\images_text_task1_val.tsv"
data_test_path = "data\\images_text_task1_test.tsv"

data_train, labels_train,train_id,train_le,n_classes = generate_data_file(data_train_path)
nb_train_samples=len(data_train)

data_val, labels_val, val_id, _, _ = generate_data_file(data_val_path)
nb_validation_samples=len(data_val)

data_test, labels_test, test_id ,_, _ = generate_data_file(data_test_path)
nb_test_samples=len(data_test)


data_train = np.expand_dims(data_train, axis=0)
data_train = preprocess_input(data_train, data_format=None)
data_train = data_train[0]

data_val = np.expand_dims(data_val, axis=0)
data_val = preprocess_input(data_val, data_format=None)
data_val = data_val[0]

model=instantiate_vgg_model(n_classes)


early_stopping_callback = callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
lr_reduce = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1,
                                            factor=0.1, min_lr=0.0001,mode='max')
checkpoint = ModelCheckpoint("models", monitor='val_acc', verbose=1,
                                save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [early_stopping_callback, lr_reduce, checkpoint]


model.fit(data_train, labels_train, batch_size=batch_size, epochs=n_epoch,verbose=1, validation_data=(data_val, labels_val),callbacks=callbacks_list)    
                        
model.save("models")


results_val=model.predict([data_val], batch_size=batch_size, verbose=1)
AUC, accu, P, R, F1, results_detail = metrics.calculate_metrics(labels_val, results_val, train_le)

result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
    "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"
print(result)
print (results_detail)

data_val=[]

data_test = np.expand_dims(data_test, axis=0)
data_test = preprocess_input(data_test, data_format=None)
data_test = data_test[0]

results_test=model.predict([data_test], batch_size=batch_size, verbose=1)
AUC, accu, P, R, F1, results_detail = metrics.calculate_metrics(labels_test, results_test, train_le)

result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
    "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"

print(result)
print (results_detail)
