# Pre-installs required.
'''
You need to install the correct pythons libraries as required from Windows'CMD:
pip install scipy
pip install pandas
...etc
'''

# import libaries
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd

# reading in data
os.chdir('C:\\Users\\Ahboy\\Desktop\\Datasets') #Change folder_path based on your PC.
rit18 = loadmat('rit18_data.mat') #file_name

# returns dictionary index for
# comment/uncomment as needed for print commands
print(rit18.keys())
print(type(rit18['train_data']) , rit18['train_data'].shape, rit18['train_labels'].shape, rit18['classes'])


xTrain = rit18['train_data']


print(xTrain[:6].shape)

