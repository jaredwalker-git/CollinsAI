#Code is running on Python 3.7.9 version


#Version 3 - Testing


# import libraries
import os
from osgeo import gdal #Refer to requirements.txt, if an error occur
import pandas as pd #pip install pandas
from scipy.io import loadmat #pip install scipy
from scipy import stats
from pyrsgis.convert import changeDimension 
import numpy as np

#Import the Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Sequential

from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

###########################################################################
def make_chips_data(image, chip_width, chip_height):

    dataset_of_chips = []
    #Finding number of chips by dimensional resolution divided by desired size
    num_of_chips_x = train_data.shape[1] // chip_width
    num_of_chips_y = train_data.shape[0] // chip_height

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
    num_of_chips_x = train_data.shape[1] // chip_width
    num_of_chips_y = train_data.shape[0] // chip_height

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

###########################################             
'''
# Directory call for user input
print("Enter Directory:")
userinput = input()
chdir = os.chdir(userinput)
'''

filepath = "C:\\Users\\Jared\\Documents\\Datasets"
os.chdir(filepath)

dataset = loadmat('rit18_data.mat')
dataset_labels = loadmat('rit18_asphalt_vegetation.mat')

#Load Training Data and Labels
train_data = dataset['train_data']
train_labels = dataset_labels['pixel_labels']

#moves bands to channel last
train_data = np.moveaxis(train_data, 0, -1)
print("Train Data shape: ", train_data.shape)

#splitting 7th band of orthomosiac from train data and load train labels
train_mask = train_data[:,:,-1]
train_data = train_data[:,:,:6]
#print("Train Mask Shape: ", train_mask.shape)


plt.imshow(train_data[:,:,5])
#plt.show()
chip_width, chip_height = (40,40)


# show mask with grid
# chips in the yellow region must be kept
fig, ax = plt.subplots()
plt.imshow(train_mask[:,:])
grid_x = np.arange(0, train_data.shape[1], chip_width)
grid_y = np.arange(0, train_data.shape[0], chip_width)
ax.set_xticks(grid_x)
ax.set_yticks(grid_y)
ax.grid(which='both')
#plt.show()

#prints number of chips 
print("Number of chips in X:", train_data.shape[1] // chip_width)
print("Number of chips in Y:", train_data.shape[0] // chip_height)
train_data_chips =   make_chips_data(train_data, chip_width, chip_height)
train_labels_softmax = make_chips_labels(train_labels, chip_width, chip_height)
numChips = train_labels_softmax.shape[0]
print("Label chips for test \n", train_labels_softmax[1,:])
print("Total Number of Chips Taken from Mask: ", numChips)

print("Train Data Chip Shape: ", train_data_chips.shape)
print("Label Data Chip Shape: ", train_labels_softmax.shape) 

num_of_chips_x = train_data.shape[1] // chip_width
num_of_chips_y = train_data.shape[0] // chip_height


#show chip 1 for proof of concept
plt.imshow(train_data_chips[1,:,:,5])
#plt.show()
#END


#######################################################################
#Create the model
model = Sequential()

#encoder (down sampling)
model.add(Input(shape = (40, 40, 6)))
layer1 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(0.001))
model.add(layer1)
model.add(Conv2D(16, kernel_size=(3, 3), strides= 1,  padding ='same', activation='relu',  data_format='channels_last', input_shape=(40,40,6)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'valid', data_format = 'channels_last'))
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'valid', data_format = 'channels_last'))
model.add(Conv2D(64, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
#decoder (up sampling)
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(UpSampling2D(size=(2,2), data_format = 'channels_last'))
model.add(Conv2D(16, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(UpSampling2D(size=(2,2), data_format = 'channels_last'))
model.add(Flatten())  #Add a “flatten” layer which prepares a vector for the fully connected layers
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))

#######################################################################

# Create a callback that saves the model's weights
save_weights = "training_weights_FINAL_R_01_BS_32.ckpt"
trn_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights, save_weights_only=True)

#######################################################################

#Create summary of our model
model.summary()
#Compile the model
model.compile(optimizer=Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(train_data_chips, train_labels_softmax, batch_size=32, epochs=200, verbose=1, shuffle=True, callbacks=[trn_callback])

#######################################################################

#Train the model
print("Weights save as file:", save_weights)