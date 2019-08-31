# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:26:00 2019

@author: GUOFENG
"""

import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers


def implt(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')

    # Set up 'ggplot' style


plt.style.use('ggplot')  # if want to use the default style, set 'classic'
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family'] = 'Arial'

# .............................................................................


data = cifar10.load_data()
(trDat, trLbl) = data[0]
(tsDat, tsLbl) = data[1]

# Convert the data into 'float32'
# Rescale the values from 0~255 to 0~1
trDat = trDat.astype('float32') / 255
tsDat = tsDat.astype('float32') / 255

# Retrieve the row size of each image
# Retrieve the column size of each image
imgrows = trDat.shape[1]
imgclms = trDat.shape[2]
channel = trDat.shape[3]

# Perform one hot encoding on the labels
# Retrieve the number of classes in this problem
trLbl = to_categorical(trLbl)
tsLbl = to_categorical(tsLbl)
num_classes = tsLbl.shape[1]

# .............................................................................

# fix random seed for reproducibility
seed = 29
np.random.seed(seed)

optmz = optimizers.RMSprop(lr=0.0001)
modelname = 'Exercise_4_1'


# define the deep learning model


def createModel():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = createModel()  # This is meant for training
modelGo = createModel()  # This is used for final testing

model.summary()

# .............................................................................


# Create checkpoint for the training
# This checkpoint performs model saving when
# an epoch gives highest testing accuracy
filepath = modelname + ".hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True,
                             mode='max')

# Log the epoch detail into csv
csv_logger = CSVLogger(modelname + '.csv')
callbacks_list = [checkpoint, csv_logger]

# .............................................................................


# Fit the model
# This is where the training starts
model.fit(trDat,
          trLbl,
          validation_data=(tsDat, tsLbl),
          epochs=200,
          batch_size=128,
          shuffle=True,
          callbacks=callbacks_list)

# ......................................................................


# Now the training is complete, we get
# another object to load the weights
# compile it, so that we can do
# final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy',
                optimizer=optmz,
                metrics=['accuracy'])

# .......................................................................


# Make classification on the test dataset
predicts = modelGo.predict(tsDat)

# Prepare the classification output
# for the classification report
predout = np.argmax(predicts, axis=1)
testout = np.argmax(tsLbl, axis=1)
labelname = ['airplane',
             'automobile',
             'bird',
             'cat',
             'deer',
             'dog',
             'frog',
             'horse',
             'ship',
             'truck']
# the labels for the classfication report


testScores = metrics.accuracy_score(testout, predout)
confusion = metrics.confusion_matrix(testout, predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores * 100))
print(metrics.classification_report(testout, predout, target_names=labelname, digits=4))
print(confusion)

# ..................................................................

import pandas as pd

records = pd.read_csv(modelname + '.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0, 0.20, 0.40, 0.60, 0.80, 1.00])
plt.title('Loss value', fontsize=12)

ax = plt.gca()
ax.set_xticklabels([])

plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
plt.title('Accuracy', fontsize=12)
plt.show()

from tensorflow.keras.utils import plot_model

plot_model(model,
           to_file=modelname + '_model.pdf',
           show_shapes=True,
           show_layer_names=False,
           rankdir='TB')
