import pandas as pd
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sub_list = ["Sub" + str(i) for i in range(2, 21)]
sub_list.remove('Sub4')

DATA_FOLDER = r'D:/NUS_TERM2_CA3/MAREA_dataset'
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, 'Processed_data')
SUBJECT_FOLDER = os.path.join(DATA_FOLDER, 'Subject Data_txt format')

full_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'Sub1_processed.csv'))
full_df = full_df.drop(full_df.columns[[0]], axis=1)

lf_df = pd.read_csv(os.path.join(SUBJECT_FOLDER, 'Sub1_LF.txt'))

print(lf_df.head())

plt.plot(lf_df['accX'])
plt.plot(lf_df['accY'])
plt.plot(lf_df['accZ'])
plt.title('Sub1')
plt.ylabel('acc')
plt.xlabel('time')
plt.legend(['accX', 'accY','accZ'], loc='upper left')
plt.savefig('sub1', dpi=300)
plt.show()

# for sub in sub_list:
#     tmp_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, sub + '_processed.csv'))
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


