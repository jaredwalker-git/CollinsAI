#Code is running on Python 3.7.9 version


#Version 3 - Testing


# import libraries
import os
from osgeo import gdal #Refer to requirements.txt, if an error occur
import pandas as pd #pip install pandas
from scipy.io import loadmat #pip install scipy
from pyrsgis.convert import changeDimension 
import numpy as np

# Directory call for user input
#print("Enter Directory:")
#userinput = input()
#chdir = os.chdir(userinput)

# NOTE: Alter filepath + name lines below to personal specified file address
filepath = "C:\\Users\\Ahboy\\Desktop\\Datasets"
os.chdir(filepath)


# Update status of running program
print("Status: Program is running correctly. (ignore-Warning Signs)")

# Loads all variables stored in the MAT-file into a simple Python data structure
rit18data = loadmat("rit18_data.mat")

trainData = rit18data['train_data']
trainLabels = rit18data['train_labels']

valData = rit18data['val_data']
valLabels = rit18data['val_labels']

#Returns data shape
print("shape=(trainData): ", trainData.shape)
print("shape=(trainLabels): ", trainLabels.shape)

print("shape=(valData): ", valData.shape)
print("shape=(valLabels): ", valLabels.shape)


#Import the Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from tensorflow.keras.losses import sparse_categorical_crossentropy 
from tensorflow.keras.optimizers import Adam

#reshape it to a 4D array
#input must have the four-dimensional shape [samples, rows, columns, channels] 
'''
will hold the raw pixel values of the image, 
in this case an image of width X, height Y, and with 6 color channels.
'''
trainData = trainData.reshape(1,9393,5642,7)
valData = valData.reshape(1,8833,6918,7)

#Converts a class vector (integers) to binary class matrix
trainLabels = tf.keras.utils.to_categorical(trainLabels)
#valLabels = tf.keras.utils.to_categorical(valLabels)

trainLabels = trainLabels.reshape(1,-1)
valLabels = valLabels.reshape(1,-1)


print("shape2=(trainData): ", trainData.shape)
print("shape2=(trainLabels): ", trainLabels.shape)

print("shape2=(valData): ", valData.shape)
print("shape2=(valLabels): ", valLabels.shape)


# Scale data
#trainData = trainData / 255
#trainLabels = trainLabels / 255


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1,  padding ='same', activation='relu',  data_format='channels_last', input_shape=(9393,5642,7)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding = 'valid'))
model.add(Conv2D(64, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding = 'valid'))
model.add(Conv2D(128, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(Flatten())
#model.add(Dense(6, activation='sigmoid'))

#create summary of our model
model.summary()

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.fit(trainData, trainLabels, validation_data=(valData, valLabels), batch_size=16, epochs=5, verbose=1, shuffle=True)

#model.fit(trainData, trainLabels, validation_data=(valData, valLabels), epochs=5)











'''
# Loads all variables stored in the MAT-file into a simple Python data structure
rit18data = loadmat("rit18_data.mat")

trainData = rit18data['train_data']
trainLabels = rit18data['train_labels']

valData = rit18data['val_data']
valLabels = rit18data['val_labels']

#Returns data shape
print("Multispectral image shape: ", trainData.shape)
print("Label array shape: ", trainLabels.shape)

print("Test data shape: ", valData.shape)
print("Test label shape: ", valLabels.shape)


# Change to 1d array from numpy array where columns are bands and rows are pixels
trainData = changeDimension(trainData)
trainLabels = changeDimension(trainLabels)

valData = changeDimension(valData)
valLabels = changeDimension(valLabels)


nBands = trainData.shape[1]


print("New Feature image shape: ", trainData.shape)
print("New Label image shape: ", trainLabels.shape)






from tensorflow import keras

# Define the parameters of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(36, activation='relu'),
    keras.layers.Dense(nBands, activation='softmax')])

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Update status of running program
print("Status: Program is training model. Please wait...")

# Run the model
model.fit(trainData, trainLabels, epochs=10, batch_size = 64)




from sklearn.metrics import confusion_matrix, precision_score, recall_score

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
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))


'''