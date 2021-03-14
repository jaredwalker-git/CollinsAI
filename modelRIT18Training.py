#Code is running on Python 3.7.9 version

# import libraries
import os 
from osgeo import gdal #Refer to requirements.txt, if an error occur
import pandas as pd #pip install pandas
from scipy.io import loadmat #pip install scipy
from pyrsgis.convert import changeDimension 
import numpy as np
from tensorflow import keras
import tensorflow as tf

# Directory call for user input
#print("Enter Directory:")
#filepath = input()
#chdir = os.chdir(filepath)

# NOTE: Alter filepath + name lines below to personal specified file address
filepath = "C:\\Users\\Jared\\Documents\\Datasets"
os.chdir(filepath)


# Update status of running program
print("Status: Program is running correctly. (ignore-Warning Signs)")

# Loads all variables stored in the MAT-file into a simple Python data structure
rit18data = loadmat("rit18_data.mat")

trainData = rit18data['train_data']
trainLabels = rit18data['train_labels']


#Returns data shape
print("Multispectral image shape: ", trainData.shape)
print("Label array shape: ", trainLabels.shape)


# Temporary minimization of data until label normalization is done
#chooseData = np.random.randint(52995306, size = 500000) #500k out of 52M random inputs is chosen

#trainData = trainData[chooseData]
#trainLabels = trainLabels[chooseData]

# Print the shape of reshaped data
print(trainData.shape, trainLabels.shape)




# Define the parameters of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(36, activation='relu'),
    keras.layers.Dense(nBands, activation='softmax')])


# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Update status of running program
print("Status: Program is training model. Please wait...")


# Create a callback that saves the model's weights
save_weights = "training_weights.ckpt"
trn_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights, save_weights_only=True)


# Run the model
model.fit(trainData, trainLabels, epochs=9, batch_size = 64, callbacks=[trn_callback])

print("Weights save as file:", save_weights)