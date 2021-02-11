from scipy.io import loadmat
import os
import numpy as np
import pandas as pd

#reading in data
os.chdir(r'C:\Users\Jared\Documents\Datasets')
rit18 = loadmat('rit18_data.mat')

#return header of each colomn
print(rit18.keys())

print(type(rit18['train_data']) , rit18['train_data'].shape, rit18['train_labels'].shape, rit18['classes'])

data = [[row.flat[0] for row in line] for line in rit18['train_data'][0]]
df_train = pd.DataFrame(data)
print(df_train)

