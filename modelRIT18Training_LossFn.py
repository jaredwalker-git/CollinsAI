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

#def band_importance(y_true, y_pred, smooth, thresh)

# NOTE: Alter filepath + name lines below to personal specified file address
filepath = "G:\OneDrive - University of Massachusetts Lowell - UMass Lowell\Github School\Dataset"
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
#Practice Example: root-Mean Square Error
def rmse(y_true, y_pred): 
    return  K.sqrt(K.mean(K.square(y_pred - y_true)))

#categorical_crossentropy

#Plot 2D Conv Weights/Filters
print("layer1 weight shape: ", layer1.get_weights()[0])
x1w = layer1.get_weights()[0][:,:,0,:]

for i in range(1,26):
    plt.subplot(5,5,i)
    plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
plt.show()

#Compile the model
model.compile(optimizer="adam", loss=rmse, metrics=["accuracy"])

# Update status of running program
print("Status: Program is training model. Please wait...")


# Create a callback that saves the model's weights
save_weights = "training_weights.ckpt"
trn_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights, save_weights_only=True)


#Train the model
model.fit(trainData, trainLabels, batch_size=16, epochs=5, verbose=1, shuffle=True, callbacks=[trn_callback])

#Plot 2D Conv Weights/Filters after training
x1w = layer1.get_weights()[0][:,:,0,:]
for i in range(1,26):
    plt.subplot(5,5,i)
    plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
plt.show()
print("Weights save as file:", save_weights)