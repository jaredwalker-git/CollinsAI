from scipy.io import loadmat
import os
import numpy as np
import pandas as pd

#reading in data

os.chdir('C:\\Users\\Jared\\Documents\\Datasets')
rit18 = loadmat('rit18_data.mat')

#returns dictionary index
#print(rit18.keys())
#print(type(rit18['train_data']) , rit18['train_data'].shape, rit18['train_labels'].shape, rit18['classes'])

xTrain = rit18['train_data']


print(chooseData)
print(xTrain[:6].shape)

