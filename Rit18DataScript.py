from scipy.io import loadmat
import os
import numpy as np
import pandas as pd

#reading in data
print("Enter Directory:")
userinput = input()
os.chdir(userinput)
rit18 = loadmat('rit18_data.mat')

#returns dictionary index
print(rit18.keys())

print(type(rit18['train_data']) , rit18['train_data'].shape, rit18['train_labels'].shape, rit18['classes'])

data = [[row.flat[0] for row in line] for line in rit18['train_data'][0]]
df_train = pd.DataFrame(data)


