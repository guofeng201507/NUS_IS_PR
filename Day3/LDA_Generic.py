# Workshop on Linear Discriminant Analysis
# Course: Problem Solving using Pattern Recognition (PSUPR)
# Written by Charles Pang 13-17 May 2019

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report, precision_score

df = pd.read_excel('Custchurn.xlsx',sheet_name = "Custchurn")
features = list(df)
X = df.drop('churn',axis=1) # DataFrame minus churn column
y = df[:]['churn']  # 1-d labelled array (~column)

# (1) Summary Statistics
sumry = np.round(df.describe().transpose(),decimals=2)
print("Data Dimension:",df.shape)
print("Summary Statistcs\n",sumry)

# (2)) distribution plots for each feature.
print("Frequency Distribution:\n")
df.hist(grid=True, figsize=(10,8))
plt.tight_layout()
plt.show()

# (2) correlation matrices (before stdze)
print('Corelation Matrix:\n',np.round(df.corr(),decimals=3))

# (3) Split Training & Testing datasets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# (4) Applying LDA
lda = LDA(n_components=2)
X_train = lda.fit(X_train, y_train)

# (5) Show the result of LDA
print("Model Priors:\n",lda.priors_) # [%churn, %no_churn]


# Get the laodings
scalings = pd.DataFrame(lda.scalings_)
scalings.insert(0,'features',features[0:len(features)-1])
print("LDA Loadings:\n",scalings)

# Get the Eigenvectors
coeff= pd.DataFrame(np.transpose(lda.coef_))
coeff.insert(0,'features',features[0:len(features)-1])
print("\nLDA Coefficients:\n",coeff)

# (6) Apply LDA on the Testset
pred=lda.predict(X_test)
print(np.unique(pred, return_counts=True))
print("\nConfusion Matrix:\n", confusion_matrix(pred, y_test))
print('\n')
print(classification_report(y_test, pred, digits=3))

