import os

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DATA_FOLDER = r'D:/NUS_TERM2_CA3/MAREA_dataset'
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, 'Processed_data')

sub_list = ["Sub" + str(i) for i in range(2, 21)]
sub_list.remove('Sub4')

full_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'Sub1_processed.csv'))
full_df = full_df.drop(full_df.columns[[0]], axis=1)

print(full_df.shape)

for sub in sub_list:
    tmp_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, sub + '_processed.csv'))
    tmp_df = tmp_df.drop(tmp_df.columns[[0]], axis=1)

    print('Loading ' + sub + '_processed.csv')
    print(tmp_df.shape)
    full_df = full_df.append(tmp_df, ignore_index=True)

print(full_df.shape)

# %%
df_tread_flat_walk = full_df[full_df['label'] == 'tread_flat_walk']
print(df_tread_flat_walk.shape)

df_tread_flat_run = full_df[full_df['label'] == 'tread_flat_run']
print(df_tread_flat_run.shape)

df_tread_slope_walk = full_df[full_df['label'] == 'tread_slope_walk']
print(df_tread_slope_walk.shape)

df_indoor_flat_walk = full_df[full_df['label'] == 'indoor_flat_walk']
print(df_indoor_flat_walk.shape)

df_indoor_flat_run = full_df[full_df['label'] == 'indoor_flat_run']
print(df_indoor_flat_run.shape)

df_rest = full_df[full_df['label'] == 'rest']
print(df_rest.shape)

df_outdoor_walk = full_df[full_df['label'] == 'outdoor_walk']
print(df_outdoor_walk.shape)

df_outdoor_run = full_df[full_df['label'] == 'outdoor_run']
print(df_outdoor_run.shape)

# %%
window_size = 256
number_columns = 13

activity_to_num_mapping = {
    "rest": 0,
    "tread_flat_walk": 1,
    "tread_flat_run": 2,
    "tread_slope_walk": 3,
    "indoor_flat_walk": 4,
    "indoor_flat_run": 5,

    "outdoor_walk": 6,
    "outdoor_run": 7
}


def reshape_df(df, window_size, number_columns):
    n_drop = df.shape[0] % window_size
    n_samples = df.shape[0] // window_size
    df = df[:-n_drop]

    label = activity_to_num_mapping.get(df.iloc[0][12])
    label_series = pd.Series([label for _ in range(n_samples)])

    return df.values.reshape(n_samples, window_size, number_columns), label_series


# %%
df_tread_flat_walk_3d, ds_tread_flat_walk_label = reshape_df(df_tread_flat_walk, window_size, number_columns)
print(df_tread_flat_walk_3d.shape)
print(ds_tread_flat_walk_label.shape)

df_tread_flat_run_3d, ds_tread_flat_run_label = reshape_df(df_tread_flat_run, window_size, number_columns)
print(df_tread_flat_run_3d.shape)
print(ds_tread_flat_run_label.shape)

df_tread_slope_walk_3d, ds_tread_slope_walk_label = reshape_df(df_tread_slope_walk, window_size, number_columns)
print(df_tread_slope_walk_3d.shape)
df_indoor_flat_walk_3d, ds_indoor_flat_walk_label = reshape_df(df_indoor_flat_walk, window_size, number_columns)
print(df_indoor_flat_walk_3d.shape)
df_indoor_flat_run_3d, ds_indoor_flat_run_label = reshape_df(df_indoor_flat_run, window_size, number_columns)
print(df_indoor_flat_run_3d.shape)
df_outdoor_walk_3d, ds_outdoor_walk_label = reshape_df(df_outdoor_walk, window_size, number_columns)
print(df_outdoor_walk_3d.shape)
df_outdoor_run_3d, ds_outdoor_run_label = reshape_df(df_outdoor_run, window_size, number_columns)
print(df_outdoor_run_3d.shape)
df_rest_3d, ds_rest_label = reshape_df(df_rest, window_size, number_columns)
print(df_rest_3d.shape)

full_df_3d = np.vstack((df_tread_flat_walk_3d, df_tread_flat_run_3d,
                        df_tread_slope_walk_3d, df_indoor_flat_walk_3d,
                        df_indoor_flat_run_3d, df_outdoor_walk_3d,
                        df_outdoor_run_3d, df_rest_3d
                        ))

print(full_df_3d.shape)

full_ds_label = ds_tread_flat_walk_label.append(ds_tread_flat_run_label)
full_ds_label = full_ds_label.append(ds_tread_slope_walk_label)
full_ds_label = full_ds_label.append(ds_indoor_flat_walk_label)
full_ds_label = full_ds_label.append(ds_indoor_flat_run_label)
full_ds_label = full_ds_label.append(ds_outdoor_walk_label)
full_ds_label = full_ds_label.append(ds_outdoor_run_label)
full_ds_label = full_ds_label.append(ds_rest_label)

print(full_ds_label.shape)

# %%
df_array = full_df_3d[:, :, :-1]
print(df_array.shape)

from keras.utils.np_utils import to_categorical

y_cat = to_categorical(full_ds_label, num_classes=8)
print(y_cat)
print(y_cat.shape)
# %%
y = y_cat
X = df_array

padding_number = 19  # 12000 - 11981

df_padding_X = df_array[0:19, :, :]
df_padding_y = y_cat[0:19, :]

print(df_padding_X.shape)
print(df_padding_y.shape)

X = np.vstack((X, df_padding_X))
y = np.vstack((y, df_padding_y))

print(X.shape)
print(y.shape)

# %%
from sklearn.model_selection import train_test_split, cross_val_score

# https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=5)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# print(X_train.describe())
# print(y_train.describe())
# %%
# ------------------------------
# https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()

batch_size = 300

# Recurrent layer
model.add(LSTM(100, batch_input_shape=(batch_size, 256, 12), return_sequences=False, dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(128))

# Fully connected layer
model.add(Dense(128, activation='relu'))


# Dropout for regularization
# model.add(Dropout(0.5))

# Output layer
model.add(Dense(8, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

STEPS = X_train.shape[0] // 20
# VALID_STEPS = validation_generator.n // 20

history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=1)

# %%
print(model.summary())

score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
# %%
model_json = model.to_json()
with open("ca3_model_v2.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights('ca3_model_v2.h5')
print("Saved model to disk")
# %%
from tensorflow.keras.models import model_from_json

# load json and create model
json_file = open('ca3_model_v2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("ca3_model_v2.h5")
print("Loaded model from disk")

# Compile the model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
print(model.summary())

score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
