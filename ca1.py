# %% md
# CA 1 Cleansing

# %%
import pandas as pd

pd.set_option('display.max_columns', 500)

df = pd.read_csv(r'D:\NUS_TERM2_CA1\application_train.csv')
df.head(10)

# %%
null_columns = df.columns[df.isnull().any()]
print(df[null_columns].isnull().sum())

# %%
df.drop(['SK_ID_CURR'], axis=1, inplace=True)

# %%

df.dropna(thresh=len(df) * 0.7, axis=1, inplace=True)
print(df.shape)

# %%
null_columns = df.columns[df.isnull().any()]
print(df[null_columns].isnull().sum())

# %%
for column in df.columns:
    if len(df[column].unique()) < 100:
        df[column] = df[column].astype('category')

cat_columns = df.select_dtypes(['category']).columns

print(cat_columns)

# %%
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

df.head(5)

# %%
null_columns = df.columns[df.isnull().any()]
for null_column in null_columns:
    df[null_column] = df[null_column].fillna(df[null_column].mean())

null_columns = df.columns[df.isnull().any()]
print(df[null_columns].isnull().sum())

# %%
# Log Regression
y = df['TARGET']
X = df.drop('TARGET', axis=1)

from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=42)
# if stratify as true, train set is (7164, 70), test set is (5379, 70)

print(X_train.shape)

print(X_test.shape)

#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Log Regression -----------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=6).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# %%
# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
# ROC
from sklearn import metrics
import matplotlib.pyplot as plt

print("Accuracy=", metrics.accuracy_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label="logreg, auc=" + str(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc=4)
plt.show()

# %%
# Naiive Bayes -------------------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Initiating the Gaussian Classifier
mod = GaussianNB()
# %%
# Training your model
mod.fit(X_train, y_train)

print("Training set score: {:.3f}".format(mod.score(X_train, y_train)))
print("Test set score: {:.3f}".format(mod.score(X_test, y_test)))

# %%
# Predicting Outcome
predicted = mod.predict(X_test)
# %%
mod.score(X_test, y_test)
# %%
# Confusion Matrix
y_pred = mod.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Naive Bayes performs well when we have multiple classes and working with text classification. Advantage of Naive Bayes algorithms are:
#
# It is simple and if the conditional independence assumption actually holds, a Naive Bayes classifier will converge quicker than discriminative models like logistic regression, so you need less training data. And even if the NB assumption doesn’t hold.
# It requires less model training time
# The main difference between Naive Bayes(NB) and Random Forest (RF) are their model size. Naive Bayes model size is low and quite constant with respect to the data. The NB models cannot represent complex behavior so it won’t get into over fitting. On the other hand, Random Forest model size is very large and if not carefully built, it results to over fitting. So, When your data is dynamic and keeps changing. NB can adapt quickly to the changes and new data while using a RF you would have to rebuild the forest every time something changes.

# !!!!!!!!!!!!!!!!!!!!!!!random forest is better than log regression , gradient boosting better than random forest

# %%
# random forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf.fit(X_train, y_train)

print("Training set score: {:.3f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.3f}".format(clf.score(X_test, y_test)))

# %%
# ROC
from sklearn import metrics
import matplotlib.pyplot as plt

print("Accuracy=", metrics.accuracy_score(y_test, y_pred))

y_pred_proba = clf.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label="random forest, auc=" + str(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc=4)
plt.show()

# %%
# GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingRegressor

params = {
    'n_estimators': 1,
    'max_depth': 5,
    'learning_rate': 1,
    'criterion': 'mse'
}

gbr = GradientBoostingRegressor(**params)
gbr.fit(X_train, y_train)

print("Training set score: {:.3f}".format(clf.score(X_train, y_train)))
print("Test set score: {:.3f}".format(clf.score(X_test, y_test)))

# %%
# Neural network

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
mlp.fit(X_train, y_train)

print("Training set score: {:.3f}".format(mlp.score(X_train, y_train)))
print("Test set score: {:.3f}".format(mlp.score(X_test, y_test)))

#%%
#cross validation score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
logreg = make_pipeline(StandardScaler(), LogisticRegression(C=6))
print(cross_val_score(logreg, X, y, cv=10))

