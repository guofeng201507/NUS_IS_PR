import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sub_list = ["Sub" + str(i) for i in range(2, 21)]
sub_list.remove('Sub4')

full_df = pd.read_csv('Sub1_processed.csv')
full_df = full_df.drop(full_df.columns[[0]], axis=1)

for sub in sub_list:
    tmp_df = pd.read_csv(sub + '_' + 'processed.csv')
    tmp_df = tmp_df.drop(tmp_df.columns[[0]], axis=1)

    full_df.append(tmp_df)

print(full_df.head())

y = full_df['label']
X = full_df.drop('label', axis=1)

print(X.corr())

from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=5)

print(X_train.shape)
print(y_train.shape)
print(X_train.describe())
print(y_train.describe())

# xgboost
print('Option 2: xgboost')
import xgboost as xgboost
import time

print('#Option 2: xgboost')
model = xgboost.XGBClassifier()
start = time.time()
model.fit(X_train, y_train)
end = time.time()
timing = end - start
print(str(timing))
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC
from sklearn import metrics
import matplotlib.pyplot as plt

print("Accuracy=", metrics.accuracy_score(y_test, y_pred))

y_pred_proba = model.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label="auc=xgboost" + str(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc=4)
plt.show()


#------------------------------
#https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()

# Embedding layer
model.add(
    Embedding(input_dim=num_words,
              input_length = training_length,
              output_dim=100,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False,
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_words, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
