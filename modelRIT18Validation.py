#Code is running on Python 3.7.9 version

# import libraries
import os
from osgeo import gdal #Refer to requirements.txt, if an error occur
import pandas as pd #pip install pandas
from scipy.io import loadmat #pip install scipy
from pyrsgis.convert import changeDimension 
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Directory call for user input
#print("Enter Directory:")
#userinput = input()
#chdir = os.chdir(userinput)

# NOTE: Alter filepath + name lines below to personal specified file address
filepath = "C:\\Users\\Jared\\Documents\\Datasets"
os.chdir(filepath)


# Update status of running program
print("Status: Program is running correctly. (ignore-Warning Signs)")

# Loads all variables stored in the MAT-file into a simple Python data structure
rit18data = loadmat("rit18_data.mat")

valData = rit18data['val_data']
valLabels = rit18data['val_labels']

#Returns data shape

print("Test data shape: ", valData.shape)
print("Test label shape: ", valLabels.shape)


# Change to 1d array from numpy array where columns are bands and rows are pixels

valData = changeDimension(valData)
valLabels = changeDimension(valLabels)

nBands = valData.shape[1]

# Reshape the data to fit format of flattened input layer
valData = valData.reshape((valData.shape[0], 1, valData.shape[1]))


# Temporary minimization of data until label normalization is done
chooseData = np.random.randint(52995306, size = 500000) #500k out of 52M random inputs is chosen

valData = valData[chooseData]
valLabels = valLabels[chooseData]


# Define the parameters of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(36, activation='relu'),
    keras.layers.Dense(19, activation='softmax')])


# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Load weights from training
trn_weights = "training_weights.ckpt"
model.load_weights(trn_weights)

# Update status of running program
print("Status: Program is running model. Please wait...")


# Predict for test data 
valPredict = model.predict(valData)
# removes first column (inputs) of valPredict
valPredict = valPredict[:,1]


# Calculate and display the error metrics
valPredict = (valPredict>0.5).astype(int)
cMatrix = confusion_matrix(valLabels, valPredict)
pScore = precision_score(valLabels, valPredict, average = None)
rScore = recall_score(valLabels, valPredict, average = None)

# this score print needs to be fixed error:TypeError: only size-1 arrays can be converted to Python scalars due to printing as float
print("Confusion matrix: for nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))


