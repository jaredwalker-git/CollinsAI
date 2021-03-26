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
from tensorflow.keras import backend as K # for custom loss
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy 
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt



# NOTE: Alter filepath + name lines below to personal specified file address
filepath = "C:\\Users\\Jared\\Documents\\Datasets"
os.chdir(filepath)


# Update status of running program
print("Status: Program is running correctly. (ignore-Warning Signs)")



'''
# Training data with ground truth loaded and broken into blobs or chips. 
'''

trainData = np.random.rand(1, 160, 160, 7)
trainLabels = np.random.rand(1, 160, 160)


# Define the parameters of the model
#encoder (down sampling)
model = Sequential()
layer1 = Conv2D(32, kernel_size=(3, 3), strides= 1,  padding ='same', activation='relu',  data_format='channels_last', input_shape=(160,160,7))
model.add(layer1)
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding = 'valid'))
model.add(Conv2D(64, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding = 'valid'))
model.add(Conv2D(128, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
#decoder (up sampling)
model.add(Conv2D(64, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(UpSampling2D(size=(2,2), data_format = 'channels_last'))
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1, padding ='same', activation='relu', data_format='channels_last'))
model.add(UpSampling2D(size=(2,2), data_format = 'channels_last'))
#model.add(Flatten())  #Add a “flatten” layer which prepares a vector for the fully connected layers
model.add(Dense(7, activation='softmax'))
model.add(Dense(1, activation='softmax'))
#Create summary of our model
model.summary()

#Create Custome Loss Function

def band_reduction(layer1Weights, bandScore): 
    #index through filters, +1 score to bandScore for index of band that has highest weight 
    

#categorical_crossentropy

#Plot 2D Conv Weights/Filters
print("layer1 weight shape: ", layer1.get_weights()[0].shape)

layer1Weights = layer1.get_weights()[0]


#This command takes filter index 0 out of the 32 filters -> (3, 3, 7)
layer1Filter1 = layer1.get_weights()[0][:,:,:,0]
print(layer1Filter1.shape)



#lets print 3 filters for concept
for j in range(0, 7):
    plt.subplot(3, 7,j+1)
    plt.imshow(layer1Weights[:,:,j,1],interpolation="nearest",cmap="gray")    

for j in range(0, 7):
    plt.subplot(3, 7,j+8)
    plt.imshow(layer1Weights[:,:,j,2],interpolation="nearest",cmap="gray")   

for j in range(0, 7):
    plt.subplot(3, 7,j+15)
    plt.imshow(layer1Weights[:,:,j,3],interpolation="nearest",cmap="gray")   

plt.show()
 

#Compile the model
model.compile(optimizer="adam", loss=[band_reduction, 'categorical_crossentropy'], metrics=["accuracy"])

# Update status of running program
print("Status: Program is training model. Please wait...")


# Create a callback that saves the model's weights
save_weights = "training_weights.ckpt"
trn_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights, save_weights_only=True)
print("Weights save as file:", save_weights)

#Train the model
model.fit(trainData, trainLabels, batch_size=16, epochs=5, verbose=1, shuffle=True, callbacks=[trn_callback])

#Plot 2D Conv Weights/Filters after training
layer1Weights = layer1.get_weights()[0]

for j in range(0, 7):
    plt.subplot(3, 7,j+1)
    plt.imshow(layer1Weights[:,:,j,1],interpolation="nearest",cmap="gray")    

for j in range(0, 7):
    plt.subplot(3, 7,j+8)
    plt.imshow(layer1Weights[:,:,j,2],interpolation="nearest",cmap="gray")   

for j in range(0, 7):
    plt.subplot(3, 7,j+15)
    plt.imshow(layer1Weights[:,:,j,3],interpolation="nearest",cmap="gray")   

plt.show()
