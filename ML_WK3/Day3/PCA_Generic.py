# Workshop on Principal Component Analysis
# Course: Problem Solving using Pattern Recognition (PSUPR)
# Written by Charles Pang 7th July 2019

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#data = pd.read_excel('CPI.xlsx',sheet_name = "CPI")
#data = data.drop(['City'],axis=1) 

# data = pd.read_excel('LoanApplicant.xlsx',sheet_name = "LoanApplicant")
data = pd.read_excel('LoanApplicant_2.xlsx',sheet_name = "LoanApplicant")
# data = pd.read_excel('LoanApplicant_reorder.xlsx',sheet_name = "LoanApplicant")

features = list(data)
colnames = np.transpose(features)

# (1) Generate Summary Statistics
print("-----------------------")
print("Data Dimensions:  ", data.shape)

sumry = data.describe().transpose()
print("Summary Statistics:\n",sumry,'\n')

# (2) Histograms Visualisation
print("Frequency Distributions:")
data.hist(grid=True, figsize=(10,6), color='blue')
plt.tight_layout()
plt.show()

# (3) correlation matrix (before stdze)
corm = data.corr().values
print('Corelation Matrix:\n',data.corr())

# (4) Correlation scatter plots (very messy for huge features)
axes = pd.plotting.scatter_matrix(data, alpha=1.0, 
                                 figsize=(15,15), diagonal='kde', s=100)
# s=dot size
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corm[i,j], (0.8, 0.8), 
                        xycoords='axes fraction', ha='center', va='center')
plt.show()

# (5) STANDARDIZE data; mean=0;sd=1
data_std = StandardScaler().fit_transform(data)

# (6) Apply PCA. Get Eigenvctors, Eigenvalues
# Loadings = Correlation coefficient betwn PC & feature 
# n_components = len(features)
n_components = 8
pca = PCA(n_components).fit(data_std)

# generate PC labels:
PCs=[]
for l in range(1,n_components+1):
    PCs.append("PC"+str(l))
    
# Get Eigenvectors & Eigenvalues
eigvec = pca.components_.transpose()
eigval = pca.explained_variance_
    
#eigval, eigvec = np.linalg.eig(corm)   

# Calculate Loadings = Eigenvector * SQRT(Eigenvalue)
print('Loading Matrix:'); loadings= np.sqrt(eigval)*eigvec
print(pd.DataFrame(loadings,columns=PCs,index=colnames),'\n')

# (7) Print out Eigenvectors
print('\nEigenvectors (Linear Coefficients):')
print(pd.DataFrame(eigvec,columns=PCs,index=colnames),'\n')

var_expln= pca.explained_variance_ratio_ * 100
eigval = -np.sort(-eigval) #descending
npc = 6 # display-1
print("Eigenvalues   :",eigval[0:npc])
print("%Explained_Var:",var_expln[0:npc])
print("%Cumulative   :",np.cumsum(var_expln[0:npc]))
print('\n')

# (8) Loadings Plot
coeff = loadings[:,0:2]
fig = plt.figure(figsize=(8,8))
plt.xlim(-1,1)
plt.ylim(-1,1)
fig.suptitle('Loading Plot',fontsize=14)
plt.xlabel('PC-1 ('+str(var_expln[0])+'%)',fontsize=12)
plt.ylabel('PC-2 ('+str(var_expln[1])+'%)',fontsize=12)

for i in range(len(coeff[:,0])):
    plt.arrow(0,0,coeff[i,0],coeff[i,1],color='r',
              alpha=0.5,head_width=0.02, head_length=0.05,length_includes_head=True)
    plt.text(coeff[i,0]*1.15,coeff[i,1]*1.15,features[i],fontsize=15,
             color='m',ha='center',va='center')

circle = plt.Circle((0, 0), 0.9999999,  color='b', fill=False)
ax = fig.gca(); ax.add_artist(circle)
plt.grid(); plt.show()


## (9) scree plot
num_vars= len(features)
fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1

plt.plot(sing_vals, eigval, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.grid(); plt.show()
