#C:\Users\guofe\Desktop\Proj\loan.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r'C:\Users\guofe\Desktop\Proj\loan.csv')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(data.head(10))