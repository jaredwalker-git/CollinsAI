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
import matplotlib.pyplot as plt
#from scipy.io import loadmat
#import numpy as np

def make_chips(image, chip_width, chip_height):

    # input shape (6, 9393, 5642)
    # output shape (num chips, 6, 160, 160)

    dataset_of_chips = []

    num_of_chips_x = train_data.shape[2] // chip_width
    num_of_chips_y = train_data.shape[1] // chip_height

    print("num chips left right:", train_data.shape[2] // chip_width)
    print("num chips up down:", train_data.shape[1] // chip_height)

    for i in range(num_of_chips_y):
        for j in range(num_of_chips_x):
            dataset_of_chips.append(train_data[:,i*chip_height:(i+1)*chip_height, j*chip_width: (j+1)*chip_width])

    return np.array(dataset_of_chips)

###  
filepath = "G:\OneDrive - University of Massachusetts Lowell - UMass Lowell\Github School\Dataset"
os.chdir(filepath)

file_path = 'rit18_data.mat'

dataset = loadmat(file_path)
#Load Training Data and Labels
train_data = dataset['train_data']
print(train_data.shape)

train_mask = train_data[-1,:,:]
train_data = train_data[:6,:,:]
train_labels = dataset['train_labels']

plt.imshow(train_data[5,:,:])
plt.show()
chip_width, chip_height = (160,160)

# show mask with grid
# chips in the yellow region must be kept
fig, ax = plt.subplots()
plt.imshow(train_mask[:,:])
grid_x = np.arange(0, train_data.shape[2], chip_width)
grid_y = np.arange(0, train_data.shape[1], chip_width)
ax.set_xticks(grid_x)
ax.set_yticks(grid_y)
ax.grid(which='both')
plt.show()

train_data_chips =   make_chips(train_data, chip_width, chip_height)
train_labels_chips = make_chips(train_labels, chip_width, chip_height)
train_mask_chips =   make_chips(train_mask, chip_width, chip_height)

print(train_data_chips.shape)

num_of_chips_x = train_data.shape[2] // chip_width
num_of_chips_y = train_data.shape[1] // chip_height

# get test locations from around the image for viewing
# make sure test chips aren't from the edges of the images
loc_1 = int(num_of_chips_x*0.4 + num_of_chips_x*num_of_chips_y*0.6)
loc_2 = int(num_of_chips_x*0.5 + num_of_chips_x*num_of_chips_y*0.2)
loc_3 = int(num_of_chips_x*0.6 + num_of_chips_x*num_of_chips_y*0.5)

show_these_chips = (loc_1, loc_2, loc_3)

for index in show_these_chips:

    plt.imshow(train_data_chips[index,5,:,:])
    plt.show()

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
#model.add(Activation('softmax'))

#######################################################################

#Create Custome Loss Function
def band_reduction(layer1Weights, bandScore): 
    #index through filters, +1 score to bandScore for index of band that has highest weight 
    return 

#This command takes filter index 0 out of the 32 filters -> (3, 3, 7)
layer1Weights = layer1.get_weights()[0]
layer1Filter1 = layer1.get_weights()[0][:,:,:,0]
print(layer1Filter1.shape)
print(layer1Weights.shape)
#lets print 3 filters for concept
for j in range(0, 6):
    plt.subplot(3, 6,j+1)
    plt.imshow(layer1Weights[:,:,j,1],interpolation="nearest",cmap="gray")    

for j in range(0, 6):
    plt.subplot(3, 6,j+8)
    plt.imshow(layer1Weights[:,:,j,2],interpolation="nearest",cmap="gray")   

for j in range(0, 6):
    plt.subplot(3, 7,j+15)
    plt.imshow(layer1Weights[:,:,j,3],interpolation="nearest",cmap="gray")   

plt.show()

#######################################################################

#Create summary of our model
model.summary()
#Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


#Train the model
model.fit(train_data_chips, train_labels_chips, validation_data=(val_data_chips, val_labels_chips), batch_size=32, epochs=5, verbose=1, shuffle=True)

#######################################################################

#Plot 2D Conv Weights/Filters after training
layer1Weights = layer1.get_weights()[0]

for j in range(0, 6):
    plt.subplot(3, 6,j+1)
    plt.imshow(layer1Weights[:,:,j,1],interpolation="nearest",cmap="gray")    

for j in range(0, 6):
    plt.subplot(3, 6,j+8)
    plt.imshow(layer1Weights[:,:,j,2],interpolation="nearest",cmap="gray")   

for j in range(0, 6):
    plt.subplot(3, 7,j+15)
    plt.imshow(layer1Weights[:,:,j,3],interpolation="nearest",cmap="gray")   

plt.show()