import os

import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DATA_FOLDER = r'D:/NUS_TERM2_CA3/MAREA_dataset'
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, 'Processed_data')

sub_list = ["Sub" + str(i) for i in range(2, 21)]
sub_list.remove('Sub4')


full_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'Sub1_processed.csv'))

full_df = full_df.drop(full_df.columns[[0]], axis=1)

# for sub in sub_list:
#     tmp_df = pd.read_csv(sub + '_' + 'processed.csv')
#     tmp_df = tmp_df.drop(tmp_df.columns[[0]], axis=1)
#
#     full_df.append(tmp_df)

print(full_df.head())

y = full_df['label']
X = full_df.drop('label', axis=1)

# print(X.corr())

from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=5)

print(X_train.shape)
print(y_train.shape)
print(X_train.describe())
print(y_train.describe())
# ------------------------------
# https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()
# Recurrent layer
model.add(LSTM(64, batch_input_shape=(8, 512,10), return_sequences=False, dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

