import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers


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

data = cifar10.load_data()  # type tuple

(train_data, train_label) = data[0]
(test_data, test_label) = data[1]

# Convert the data into 'float32'
# Rescale the values from 0~255 to 0~1
train_data = train_data.astype('float32') / 255  # Training data
test_data = test_data.astype('float32') / 255  # Test data

print(train_data.shape)
print(test_data.shape)

# Retrieve the row size of each image
# Retrieve the column size of each image
imgrows = train_data.shape[1]
imgclms = train_data.shape[2]
channel = train_data.shape[3]

# For deep learning training and testing, the data must be in the
# form of (no_sample, row, clm, channel)

# Perform one hot encoding on the labels
# Retrieve the number of classes in this problem
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
num_classes = test_label.shape[1]
print(num_classes)

# .............................................................................

# fix random seed for reproducibility
seed = 29
np.random.seed(seed)

optmz = optimizers.RMSprop(lr=0.0001)
modelname = 'wks2_3'


# define the deep learning model
def createModel():
    model = Sequential()

    model.add(Conv2D(32, (3, 3),
                     input_shape=(imgrows, imgclms, channel),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optmz,
                  metrics=['accuracy'])

    return model

    # Setup the models


model = createModel()  # This is meant for training
modelGo = createModel()  # This is used for final testing

model.summary()

# Create checkpoint for the training
# This checkpoint performs model saving when
# an epoch gives highest testing accuracy
filepath = modelname + ".hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='min')

# Log the epoch detail into csv
csv_logger = CSVLogger(modelname + '.csv')
callbacks_list = [checkpoint, csv_logger]

# Fit the model
# This is where the training starts
model.fit(train_data,
          train_label,
          validation_data=(test_data, test_label),
          epochs=100,
          batch_size=128,
          shuffle=True,
          callbacks=callbacks_list)

# Now the training is complete, we get
# another object to load the weights
# compile it, so that we can do
# final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# .......................................................................


# Make classification on the test dataset
predicts = modelGo.predict(test_data)
predout = np.argmax(predicts, axis=1)
testout = np.argmax(test_label, axis=1)

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

# Prepare the classification output
# for the classification report
testScores = metrics.accuracy_score(testout, predout)
confusion = metrics.confusion_matrix(testout, predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores * 100))
print(metrics.classification_report(testout, predout, target_names=labelname, digits=4))
print(confusion)
# ..................................................................

# Plot the training output

import pandas as pd

records = pd.read_csv(modelname + '.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.yticks([0.00, 0.60, 0.70, 0.80])
plt.title('Loss value', fontsize=12)
ax = plt.gca()
ax.set_xticklabels([])
plt.subplot(212)
plt.plot(records['val_acc'])
plt.yticks([0.5, 0.6, 0.7, 0.8])
plt.title('Accuracy', fontsize=12)
plt.show()
