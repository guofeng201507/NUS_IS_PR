#C:\Users\guofe\Desktop\Proj\loan.csv

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# data = pd.read_csv(r'C:\Users\guofe\Desktop\Proj\loan.csv')
#
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
# print(data.head(10))

import pandas as pd
import numpy as np

# artificial data
# ====================================
np.random.seed(0)
df = pd.DataFrame(np.random.randn(10,5), columns=list('ABCDE'))
df[df < 0] = 0

# ====================================
df.drop([col for col, val in df.sum().iteritems() if val > 3], axis=1, inplace=True)

print(df.head())