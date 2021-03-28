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

#Import the Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

###########################################################################
#from scipy.io import loadmat
#import numpy as np

def make_chips(image, chip_width, chip_height):

    # input shape (6, 9393, 5642)
    # output shape (num chips, 6, 160, 160)

    dataset_of_chips = []

    num_of_chips_x = val_data.shape[2] // chip_width
    num_of_chips_y = val_data.shape[1] // chip_height

    print("num chips left right:", val_data.shape[2] // chip_width)
    print("num chips up down:", val_data.shape[1] // chip_height)

    for i in range(num_of_chips_y):
        for j in range(num_of_chips_x):
            dataset_of_chips.append(val_data[:,i*chip_height:(i+1)*chip_height, j*chip_width: (j+1)*chip_width])

    return np.array(dataset_of_chips)

###  
filepath = "C:\\Users\\Jared\\Documents\\Datasets"
os.chdir(filepath)
file_path = 'rit18_data.mat'
dataset = loadmat(file_path)


chip_width, chip_height = (160,160)


#Load Validation Data and Labels
val_data = dataset['val_data']

val_mask = val_data[-1]
val_data = val_data[:6]
val_labels = dataset['val_labels']

val_data_chips =   make_chips(val_data, chip_width, chip_height)
val_labels_chips = make_chips(val_labels, chip_width, chip_height)
val_mask_chips =   make_chips(val_mask, chip_width, chip_height)

#Load Test Data
test_data = dataset['test_data']
test_mask = test_data[-1]
test_data = test_data[:6]

test_data_chips =   make_chips(test_data, chip_width, chip_height)
test_mask_chips =   make_chips(test_mask, chip_width, chip_height)


#INPUT_Image = test_data_chips.reshape(2030,6,160,160)
#print("INPUT_Image REsults:", INPUT_Image)


#INPUT_Image = test_data_chips.reshape(6,160,160)
#print("train_data_chips REsults:", INPUT_Image.shape)

#######################################################################
#Create the model
model = Sequential()

#encoder (down sampling)
layer1 = (Conv2D(16, kernel_size=(3, 3), strides= 1,  padding ='same', activation='relu',  data_format='channels_first', input_shape=(6,160,160)))
model.add(layer1)
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding = 'valid', data_format = 'channels_first'))
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding = 'valid', data_format = 'channels_first'))
model.add(Conv2D(64, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_first'))
#decoder (up sampling)
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_first'))
model.add(UpSampling2D(size=(2,2), data_format = 'channels_first'))
model.add(Conv2D(16, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_first'))
model.add(UpSampling2D(size=(2,2), data_format = 'channels_first'))
#model.add(Flatten())  #Add a “flatten” layer which prepares a vector for the fully connected layers
model.add(Dense(160, activation='softmax'))
model.add(Conv2D(6, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_first'))

#######################################################################

#Create Custome Loss Function
def band_reduction(layer1Weights, bandScore): 
    #index through filters, +1 score to bandScore for index of band that has highest weight 
    return 




#######################################################################

#Create summary of our model
model.summary()
#Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#######################################################################


#Load weights from training
trn_weights = "training_weights.ckpt"
model.load_weights(trn_weights)

# Update status of running program
print("Status: Program is running model. Please wait...")


# Predict for test data 
valPredict = model.predict(val_data_chips)
# removes first column (inputs) of valPredict
valPredict = valPredict[:,1]


# Calculate and display the error metrics
valPredict = (valPredict>0.5).astype(int)
cMatrix = confusion_matrix(val_labels_chips, valPredict)
pScore = precision_score(val_labels_chips, valPredict, average = None)
rScore = recall_score(val_labels_chips, valPredict, average = None)

# this score print needs to be fixed error:TypeError: only size-1 arrays can be converted to Python scalars due to printing as float
print("Confusion matrix: for nodes\n", cMatrix)
print("\nP-Score: %d, R-Score: %d" % (pScore, rScore))


