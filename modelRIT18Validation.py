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

def make_chips_data(image, chip_width, chip_height):

    dataset_of_chips = []
    #Finding number of chips by dimensional resolution divided by desired size
    num_of_chips_x = val_data.shape[1] // chip_width
    num_of_chips_y = val_data.shape[0] // chip_height

    for i in range(num_of_chips_y):
        for j in range(num_of_chips_x):
            a = i*chip_height
            b = j*chip_width
            #Keep the pixels within the mask chips train_mask[:,:]
            if train_mask[a,b] != train_labels[a,b]:
                dataset_of_chips.append(image[i*chip_height:(i+1)*chip_height, j*chip_width: (j+1)*chip_width,:])
            else:
                continue
    return np.array(dataset_of_chips)

#seperate function for label chips due to difference in dimensionality
def make_chips_labels(image, chip_width, chip_height):

    numGoodChips = 0
    dataset_of_chips = []
    #Finding number of chips by dimensional resolution divided by desired size
    num_of_chips_x = val_data.shape[1] // chip_width
    num_of_chips_y = val_data.shape[0] // chip_height

    for i in range(num_of_chips_y):
        for j in range(num_of_chips_x):
            a = i*chip_height
            b = j*chip_width
            #Keep the pixels within the mask chips train_mask[:,:]
            if train_mask[a,b] != train_labels[a,b]:
                dataset_of_chips.append(image[i*chip_height:(i+1)*chip_height, j*chip_width: (j+1)*chip_width])
                numGoodChips = numGoodChips + 1
            else:
                continue
    #turn chips into numpy array for upcoming computation          
    dataset_of_chips = np.array(dataset_of_chips)
    train_labels_chipsGen = []

    for i in range(numGoodChips):
        train_labels_chipsGen.append(stats.mode(dataset_of_chips[i, :, :], axis = None))

    train_labels_chipsGen = np.array(train_labels_chipsGen)
    train_labels_chipsGen = train_labels_chipsGen[:,0,:]
    print("Label Data Chip Shape After Chip Generalization: ", train_labels_chipsGen.shape)
    train_labels_softmax = []

    for i in range(numGoodChips):
        if train_labels_chipsGen[i,:] == 0:
            train_labels_softmax.append([1, 0, 0, 0])
        elif train_labels_chipsGen[i,:] == 1:
            train_labels_softmax.append([0, 1, 0, 0])
        elif train_labels_chipsGen[i,:] == 2:
            train_labels_softmax.append([0, 0, 1, 0])
        elif train_labels_chipsGen[i,:] == 3:
            train_labels_softmax.append([0, 0, 0, 1])
        else:
            continue

    return np.array(train_labels_softmax)

###  
filepath = "G:\OneDrive - University of Massachusetts Lowell - UMass Lowell\Github School\Dataset"
os.chdir(filepath)
file_path = 'rit18_data.mat'
dataset = loadmat(file_path)


chip_width, chip_height = (160,160)


#Load Validation Data and Labels
val_data = dataset['val_data']

val_mask = val_data[-1]
val_data = val_data[:6]
val_labels = dataset['val_labels']

val_data_chips =   make_chips_data(val_data, chip_width, chip_height)
val_labels_chips = make_chips_labels(val_labels, chip_width, chip_height)
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

#Input layer for all layers and lambda split 
inputsX =  Input(shape = (40, 40, 6), batch_size = None) 

outputDenseLayers = []
importance_weights = []

#Creating split input and dense layer so each layer creates a weight for a single band -> must enumerate this for each band so that weights can be pulled by variable name
#Band 1
inputDenseLayer1 = Lambda(lambda x: x[:, :, :, 0:1], input_shape = (40, 40, 6))(inputsX)
DenseLayer1 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(0.001))
Out1 = DenseLayer1(inputDenseLayer1)
outputDenseLayers.append(Out1)

#Band 2
inputDenseLayer2 = Lambda(lambda x: x[:, :, :, 1:2], input_shape = (40, 40, 6))(inputsX)
DenseLayer2 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(0.001))
Out2 = DenseLayer2(inputDenseLayer2)
outputDenseLayers.append(Out2)

#Band 3
inputDenseLayer3 = Lambda(lambda x: x[:, :, :, 2:3], input_shape = (40, 40, 6))(inputsX)
DenseLayer3 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(0.001))
Out3 = DenseLayer3(inputDenseLayer3)
outputDenseLayers.append(Out3)

#Band 4
inputDenseLayer4 = Lambda(lambda x: x[:, :, :, 3:4], input_shape = (40, 40, 6))(inputsX)
DenseLayer4 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(0.001))
Out4 = DenseLayer4(inputDenseLayer4)
outputDenseLayers.append(Out4)

#Band 5
inputDenseLayer5 = Lambda(lambda x: x[:, :, :, 4:5], input_shape = (40, 40, 6))(inputsX)
DenseLayer5 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(0.001))
Out5 = DenseLayer5(inputDenseLayer5)
outputDenseLayers.append(Out5)

#Band 6
inputDenseLayer6 = Lambda(lambda x: x[:, :, :, 5:6], input_shape = (40, 40, 6))(inputsX)
DenseLayer6 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(0.001))
Out6 = DenseLayer6(inputDenseLayer6)
outputDenseLayers.append(Out6)

#Now to concatenate the dense layers
classifierInput = Concatenate(axis = 2)(outputDenseLayers)

#############################################################################################################

#Encoder (down sampling)
C1 = Conv2D(16, kernel_size=(3, 3), strides= 1,  padding ='same', activation='relu',  data_format='channels_last', input_shape=(40,40,6))(classifierInput)
P1 = MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'valid', data_format = 'channels_last')(C1)
C2 = Conv2D(32, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last')(P1)
P2 = MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'valid', data_format = 'channels_last')(C2)
C3 = Conv2D(64, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last')(P2)
#Decoder (up sampling)
C4 = Conv2D(32, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last')(C3)
U1 = UpSampling2D(size=(2,2), data_format = 'channels_last')(C4)
C5 = Conv2D(16, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last')(U1)
U2 = UpSampling2D(size=(2,2), data_format = 'channels_last')(C5)
flattenLayer = Flatten()(U2)  #Add a “flatten” layer which prepares a vector for the fully connected layers
fullyConnected = Dense(16, activation='relu')(flattenLayer)
output = Dense(3, activation='softmax')(fullyConnected)

#######################################################################


#Create summary of our model
model.summary()
#Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

#######################################################################

#Load weights from training
trn_weights = "training_weights_FINAL.ckpt"
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


