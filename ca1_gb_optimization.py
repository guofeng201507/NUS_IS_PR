# %% md
# CA 1 Cleansing

#%%
import pandas as pd

pd.set_option('display.max_columns', 500)

df = pd.read_csv(r'D:\NUS_TERM2_CA1\application_train.csv')
df.head(10)

df.dropna(thresh=len(df) * 0.7, axis=1, inplace=True)

df.drop(['SK_ID_CURR'], axis=1, inplace=True)

# Label Encoding for column with 2 or less values
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le_count = 0

# Iterate through the columns
for col in df:
    if df[col].dtype == 'object':

        if len(list(df[col].unique())) <= 2:
            le.fit(df[col])
            df[col] = le.transform(df[col])

            le_count += 1

print('%d columns were label encoded.' % le_count)

# One -hot encoding for categorical data
df = pd.get_dummies(df)
df.shape

# Fill missing value
null_columns = df.columns[df.isnull().any()]
for null_column in null_columns:
    df[null_column] = df[null_column].fillna(df[null_column].mean())

null_columns = df.columns[df.isnull().any()]
print(df[null_columns].isnull().sum())

# log transform
import numpy as np

df['AMT_INCOME_TOTAL_LOG'] = np.log(df['AMT_INCOME_TOTAL'])
df['AMT_CREDIT_LOG'] = np.log(df['AMT_CREDIT'])
df['AMT_ANNUITY_LOG'] = np.log(df['AMT_ANNUITY'])
df['AMT_GOODS_PRICE_LOG'] = np.log(df['AMT_GOODS_PRICE'])
df.drop(['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE'], axis=1, inplace=True)

df['YEARS_BIRTH_LOG'] = np.log(df['DAYS_BIRTH'].abs() / 365)
df.loc[df.DAYS_EMPLOYED >= 0, 'DAYS_EMPLOYED'] = np.NaN
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].mean())
df['YEARS_EMPLOYED_LOG'] = np.log(df['DAYS_EMPLOYED'].abs() / 365)

df.loc[df.DAYS_REGISTRATION >= 0, 'DAYS_REGISTRATION'] = np.NaN
df['DAYS_REGISTRATION'] = df['DAYS_REGISTRATION'].fillna(df['DAYS_REGISTRATION'].mean())
df['YEARS_REGISTRATION_LOG'] = np.log(df['DAYS_REGISTRATION'].abs() / 365)

df.loc[df.DAYS_ID_PUBLISH >= 0, 'DAYS_ID_PUBLISH'] = np.NaN
df['DAYS_ID_PUBLISH'] = df['DAYS_ID_PUBLISH'].fillna(df['DAYS_ID_PUBLISH'].mean())
df['YEARS_ID_PUBLISH_LOG'] = np.log(df['DAYS_ID_PUBLISH'].abs() / 365)

df.loc[df.DAYS_LAST_PHONE_CHANGE >= 0, 'DAYS_LAST_PHONE_CHANGE'] = np.NaN
df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].fillna(df['DAYS_LAST_PHONE_CHANGE'].mean())
df['YEARS_LAST_PHONE_CHANGE_LOG'] = np.log(df['DAYS_LAST_PHONE_CHANGE'].abs() / 365)

df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE'], axis=1,
        inplace=True)

df.shape


y = df['TARGET']
X = df.drop('TARGET', axis=1)

from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=42)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
normalizer.fit(X_train)

X_train = normalizer.transform(X_train)
X_test = normalizer.transform(X_test)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# # random oversampling
# # ros = RandomOverSampler(random_state=0)
# ros = SMOTE(ratio='auto', kind='regular')
# X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
# # using Counter to display results of naive oversampling
# from collections import Counter
# print(sorted(Counter(y_resampled).items()))

u_ros = RandomUnderSampler(random_state=0)
# X_under_resampled, y_under_resampled = u_ros.fit_resample(X_train, y_train)
X_resampled, y_resampled = u_ros.fit_resample(X_train, y_train)
# using Counter to display results of naive oversampling
from collections import Counter
print(sorted(Counter(y_resampled).items()))
# print(sorted(Counter(y_under_resampled).items()))

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

parameters = {
    # "loss":["deviance"],
    "learning_rate": [0.1, 0.15, 0.2, 0.3]
    # "min_samples_split": np.linspace(0.1, 0.5, 12),
    # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    }

clf = GridSearchCV(GradientBoostingClassifier(), parameters)

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.best_params_)

y_pred = clf.predict(X_test)

# %%


