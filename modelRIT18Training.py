#Code is running on Python 3.7.9 version


#Version 4 - Training


# import libraries
import os
import pandas as pd #pip install pandas
from scipy.io import loadmat #pip install scipy
from scipy import stats
import numpy as np
from pprint import pprint
import math


#Import the Modules
import tensorflow as tf
from tensorflow import keras

''' Linux
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, Input, Concatenate, Lambda, Layer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.losses import categorical_crossentropy 
from tensorflow.python.keras.optimizers import Adam, SGD
#import tensorflow.keras.backend as K
'''

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, Input, Concatenate, Lambda, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K


import matplotlib.pyplot as plt

###########################################################################
#hyperparameters
regularizer_coeff = 0.1 #replace with value of choice, recommended -> 0 to 1
epoch_num = 5  #replace with integer
chip_width, chip_height = (40,40)   #replace integers within parenthesis 
learningRate = 0.0001 #replace with value of choice - Best value from testing = 0.00001 -> typical values 0.001, 0.0001, 0.00001
batchSize = 32 #replace with value of choice -> 32 used in testing due to hardware limitations
saveFreq = 1 #replace with integer of choice -> 50 used during testing


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
            if train_mask[a,b] != train_data[a,b,1]:
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
            if train_mask[a,b] != train_data[a,b,1]:
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
            train_labels_softmax.append([1, 0, 0])
        elif train_labels_chipsGen[i,:] == 1:
            train_labels_softmax.append([0, 1, 0])
        elif train_labels_chipsGen[i,:] == 2:
            train_labels_softmax.append([0, 0, 1])
        else:
            continue

    return np.array(train_labels_softmax)

###########################################     
# Directory call for user input
userinput = input("Enter Directory for Datasets: ")
os.chdir(userinput)
print("Filepath is set... Please wait")

dataset = loadmat('rit18_data.mat')
dataset_labels = loadmat('train_labels.mat')

#Load Training Data and Labels
train_data = dataset['train_data']
train_labels = dataset_labels['relabeled_training']


#Moves bands to channel last
train_data = np.moveaxis(train_data, 0, -1)
print("Train Data shape: ", train_data.shape)

#Splitting 7th band of orthomosiac from train data and load train labels
train_mask = train_data[:,:,-1]
train_data = train_data[:,:,:6]


plt.imshow(train_data[:,:,5])
plt.show()


#Show mask with grid
#Chips in the yellow region must be kept
fig, ax = plt.subplots()
plt.imshow(train_mask[:,:])
grid_x = np.arange(0, train_data.shape[1], chip_width)
grid_y = np.arange(0, train_data.shape[0], chip_width)
ax.set_xticks(grid_x)
ax.set_yticks(grid_y)
ax.grid(which='both')
plt.show()

#Prints number of chips 
print("Number of chips in X:", train_data.shape[1] // chip_width)
print("Number of chips in Y:", train_data.shape[0] // chip_height)
train_data_chips =   make_chips_data(train_data, chip_width, chip_height)
train_labels_softmax = make_chips_labels(train_labels, chip_width, chip_height)
numChips = train_labels_softmax.shape[0]
print("Total Number of Chips Taken from Mask: ", numChips)

print("Train Data Chip Shape: ", train_data_chips.shape)
print("Label Data Chip Shape: ", train_labels_softmax.shape)

num_of_chips_x = train_data.shape[1] // chip_width
num_of_chips_y = train_data.shape[0] // chip_height


#Show chip 1 for proof of concept
chip = np.random.randint(0, train_data_chips.shape[0])
plt.imshow(train_data_chips[chip,:,:,5])
plt.show()



#######################################################################
#Input layer for all layers and lambda split 
inputsX =  Input(shape = (40, 40, 6), batch_size = None) 

outputDenseLayers = []


#Creating split input and dense layer so each layer creates a weight for a single band -> must enumerate this for each band so that weights can be pulled by variable name
#Band 1
inputDenseLayer1 = Lambda(lambda x: x[:, :, :, 0:1], input_shape = (40, 40, 6))(inputsX)
DenseLayer1 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(regularizer_coeff))
Out1 = DenseLayer1(inputDenseLayer1)
outputDenseLayers.append(Out1)

#Band 2
inputDenseLayer2 = Lambda(lambda x: x[:, :, :, 1:2], input_shape = (40, 40, 6))(inputsX)
DenseLayer2 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(regularizer_coeff))
Out2 = DenseLayer2(inputDenseLayer2)
outputDenseLayers.append(Out2)

#Band 3
inputDenseLayer3 = Lambda(lambda x: x[:, :, :, 2:3], input_shape = (40, 40, 6))(inputsX)
DenseLayer3 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(regularizer_coeff))
Out3 = DenseLayer3(inputDenseLayer3)
outputDenseLayers.append(Out3)

#Band 4
inputDenseLayer4 = Lambda(lambda x: x[:, :, :, 3:4], input_shape = (40, 40, 6))(inputsX)
DenseLayer4 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(regularizer_coeff))
Out4 = DenseLayer4(inputDenseLayer4)
outputDenseLayers.append(Out4)

#Band 5
inputDenseLayer5 = Lambda(lambda x: x[:, :, :, 4:5], input_shape = (40, 40, 6))(inputsX)
DenseLayer5 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(regularizer_coeff))
Out5 = DenseLayer5(inputDenseLayer5)
outputDenseLayers.append(Out5)

#Band 6
inputDenseLayer6 = Lambda(lambda x: x[:, :, :, 5:6], input_shape = (40, 40, 6))(inputsX)
DenseLayer6 = Dense(1, activation = None, kernel_regularizer = tf.keras.regularizers.l1(regularizer_coeff))
Out6 = DenseLayer6(inputDenseLayer6)
outputDenseLayers.append(Out6)

#Now to concatenate the dense layers
classifierInput = Concatenate(axis = 3)(outputDenseLayers)

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

#Create a callback that saves the model's weights, we will create a checkpoint every 50 epochs
iterations_per_epoch = math.ceil(numChips/batchSize)
print("Iterations per Epoch: ", iterations_per_epoch)
save_weights = input("Enter Directory for Saving Weights: ") + "\\train_{epoch:04d}.ckpt"
print("Filepath is set... Please wait")

trn_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights, save_weights_only=True, save_freq = saveFreq * iterations_per_epoch)
model = tf.keras.Model(inputs = [inputsX], outputs = output)
model.save_weights(save_weights.format(epoch=0))
model.summary()
model.compile(optimizer=Adam(lr = learningRate), loss='categorical_crossentropy', metrics=['accuracy'])


def Band_Importance(band_weights):
    def _custom_loss():
        #abs value was to minimize more aggressively
        loss = abs(tf.norm(band_weights[0])) + abs(tf.norm(band_weights[1])) + abs(tf.norm(band_weights[2])) + abs(tf.norm(band_weights[3])) + abs(tf.norm(band_weights[4])) + abs(tf.norm(band_weights[5]))
        return loss
    return _custom_loss


#pulling weights to print
band_weights = []
band_weights.append(DenseLayer1.kernel)
band_weights.append(DenseLayer2.kernel)
band_weights.append(DenseLayer3.kernel)
band_weights.append(DenseLayer4.kernel)
band_weights.append(DenseLayer5.kernel)
band_weights.append(DenseLayer6.kernel)
#model.add_loss(Band_Importance(band_weights))
pprint(band_weights)

history = model.fit(train_data_chips, train_labels_softmax, batch_size= batchSize, epochs= epoch_num, verbose=1, shuffle=True, callbacks=[trn_callback])

########### ADDED to print graphs
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.clf()
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy [%]')
plt.xlabel('Numbers of Training Epochs')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Accuracy [%]')
plt.xlabel('Numbers of Training Epochs')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
##########################################################

#print importance post train
pprint(band_weights)

print("Weights save as file:", save_weights)
