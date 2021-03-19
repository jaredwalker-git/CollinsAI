#Code is running on Python 3.7.9 version

# import libraries
import os
from osgeo import gdal #Refer to requirements.txt, if an error occur
import pandas as pd #pip install pandas
from scipy.io import loadmat #pip install scipy
from pyrsgis.convert import changeDimension 
import numpy as np

nBands = 7 

# Directory call for user input
#print("Enter Directory:")
#userinput = input()
#chdir = os.chdir(userinput)

# NOTE: Alter filepath + name lines below to personal specified file address
filepath = "G:\OneDrive - University of Massachusetts Lowell - UMass Lowell\Github School\Dataset"
os.chdir(filepath)


# Update status of running program
print("Status: Program is running correctly. (ignore-Warning Signs)")

# Loads all variables stored in the MAT-file into a simple Python data structure
rit18data = loadmat("rit18_data.mat")
rit18labels = loadmat("rit18_asphalt_vegetation")

trainData = rit18data['train_data']
trainLabels = rit18labels['pixel_labels']
trainLabels1 = rit18labels['pixel_labels']
trainLabels2 = rit18labels['pixel_labels']
trainLabels3 = rit18labels['pixel_labels']
trainLabels4 = rit18labels['pixel_labels']
trainLabels5 = rit18labels['pixel_labels']
trainLabels6 = rit18labels['pixel_labels']

trainLabels = np.dstack((trainLabels,trainLabels1))
trainLabels = np.dstack((trainLabels,trainLabels2))
trainLabels = np.dstack((trainLabels,trainLabels3))
trainLabels = np.dstack((trainLabels,trainLabels4))
trainLabels = np.dstack((trainLabels,trainLabels5))
trainLabels = np.dstack((trainLabels,trainLabels6))

valData = rit18data['val_data']
valLabels = rit18data['val_labels']

#Returns data shape
print("Multispectral image shape: ", trainData.shape)
print("Label array shape: ", trainLabels.shape)

print("Test data shape: ", valData.shape)
print("Test label shape: ", valLabels.shape)

trainData = np.array_split(trainData, 80, axis = 1)
trainLabels = np.array_split(trainLabels, 80, axis = 0)

'''
for i in trainData
    while i < 1000
        trainDataShort(i:) = trainData(i:)
'''
print("Multispectral image shape: ", trainData[8].shape)
print("Label array shape: ", trainLabels[8].shape)

'''
# Change to 1d array from numpy array where columns are bands and rows are pixels
trainData = changeDimension(trainData)
trainLabels = changeDimension(trainLabels)

valData = changeDimension(valData)
valLabels = changeDimension(valLabels)



print("New Feature image shape: ", trainData.shape)
print("New Label image shape: ", trainLabels.shape)



# Normalise the data
#trainData = trainData / 255.0
#xTest = xTest / 255.0
#featuresHyderabad = featuresHyderabad / 255.0

#Reshape the data to fit format of flattened input layer
trainData = trainData.reshape((trainData.shape[0], 1, trainData.shape[1]))
valData = valData.reshape((valData.shape[0], 1, valData.shape[1]))

# Temporary minimization of data until label normalization is done
chooseData = np.random.randint(52995306, size = 500000) #500k out of 52M random inputs is chosen


trainData = trainData[chooseData]
trainLabels = trainLabels[chooseData]

valData = valData[chooseData]
valLabels = valLabels[chooseData]
trainData.shape
'''
trainData[8] = trainData[8].reshape(1,118,5642,7)
trainLabels[8] = trainLabels[8].reshape(1,118,5642,7)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

'''
#Converts a class vector (integers) to binary class matrix
trainLabels[8] = tf.keras.utils.to_categorical(trainLabels[8])
#valLabels = tf.keras.utils.to_categorical(valLabels)

labelShape = trainLabels[8].shape
res = int(''.join(map(str, labelShape)))

trainLabels[8] = trainLabels[8].reshape(1, res)
valLabels[8] = valLabels[8].reshape(1, -1)
'''

# Print the shape of reshaped data
print(trainData[8].shape, trainLabels[8].shape)



# Define the parameters of the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), strides= 1,  padding ='same', activation='relu',  data_format='channels_last', input_shape=(118,5642,7)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding = 'valid'))
model.add(Conv2D(64, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(UpSampling2D(size=(2,2), data_format = 'channels_last'))
model.add(Dense(7, activation='sigmoid'))

model.summary()

'''
model = keras.models([
    
    keras.layers.Dense(36, activation='relu'),
    keras.layers.Dense(19, activation='softmax')])
'''

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Update status of running program
print("Status: Program is training model. Please wait...")

# Run the model
model.fit(trainData[8], trainLabels[8], epochs=5, batch_size = 64)

import shap 

from sklearn.metrics import confusion_matrix, precision_score, recall_score

importance = shap.DeepExplainer(model, trainData)

# Predict for test data 
valPredict = model.predict(valData)
# removes first column (inputs) of valPredict
valPredict = valPredict[:,1]




# Calculate and display the error metrics
valPredict = (valPredict>0.5).astype(int)
cMatrix = confusion_matrix(valLabels, valPredict)
pScore = precision_score(valLabels, valPredict, average = None)
rScore = recall_score(valLabels, valPredict, average = None)

print("Confusion matrix: for nodes\n", cMatrix)
#print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))


