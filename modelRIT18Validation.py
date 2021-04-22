#Code is running on Python 3.7.9 version


#Version 4 - Validation


# import libraries
import os
import pandas as pd #pip install pandas
from scipy.io import loadmat #pip install scipy
from scipy import stats
import numpy as np
from pprint import pprint
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from beautifultable import BeautifulTable, BTRowCollection, BTColumnCollection
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, colors
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, Input, Concatenate, Lambda, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


###########################################################################

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
            if val_mask[a,b] != val_data[a,b,1]:
                dataset_of_chips.append(image[i*chip_height:(i+1)*chip_height, j*chip_width: (j+1)*chip_width,:])
            else:
                continue
    return np.array(dataset_of_chips)

#seperate function for label chips due to difference in dimensionality
def make_chips_labels(image, chip_width, chip_height):

    numGoodChips = 0
    dataset_of_chips = []
    #Finding number of chips by dimensional resolution divided by desired size
    num_of_chips_x = val_labels.shape[1] // chip_width
    num_of_chips_y = val_labels.shape[0] // chip_height

    for i in range(num_of_chips_y):
        for j in range(num_of_chips_x):
            a = i*chip_height
            b = j*chip_width
            #Keep the pixels within the mask chips train_mask[:,:]
            if val_mask[a,b] != val_data[a,b,1]:
                dataset_of_chips.append(image[i*chip_height:(i+1)*chip_height, j*chip_width: (j+1)*chip_width])
            else:
                continue
    
    #turn chips into numpy array for upcoming computation          
    dataset_of_chips = np.array(dataset_of_chips)
    val_labels_chipsGen = []

    #generalize chips to single label
    for i in range(dataset_of_chips.shape[0]):
        val_labels_chipsGen.append(stats.mode(dataset_of_chips[i, :, :], axis = None))

    val_labels_chipsGen = np.array(val_labels_chipsGen)
    val_labels_chipsGen = val_labels_chipsGen[:,0,:]
    print("Label Data Chip Shape After Chip Generalization: ", val_labels_chipsGen.shape)
    val_labels_chips = OneHotEncoder().fit_transform(val_labels_chipsGen).toarray()
    return np.array(val_labels_chips)

def make_image_from_chips(chips, chip_width, chip_height):
    #Finding number of chips by dimensional resolution divided by desired size
    num_of_chips_x = val_labels.shape[1] // chip_width
    num_of_chips_y = val_labels.shape[0] // chip_height
    image = [] 
    chip_index = val_labels_chips.shape[0] - 1

    while chip_index >= 0:
        for i in range(num_of_chips_y):
            row = []
            for j in range(num_of_chips_x):
                a = i*chip_height
                b = j*chip_width
                #Keep the pixels within the mask chips train_mask[:,:]
                if val_mask[a,b] != val_data[a,b,1]:
                    row.append(chips[chip_index, :].argmax())
                else:
                    row.append(0)
                chip_index = chip_index - 1
            image.append(row)
    return(image)
#####################################################################################  
filepath = "C:\\Users\\Jared\\Documents\\Datasets"
os.chdir(filepath)
file_path = 'rit18_data.mat'
dataset = loadmat(file_path)

labelset = loadmat('val_labels.mat')


chip_width, chip_height = (40,40)


#Load Validation Data and Labels and move channels_last
val_data = dataset['val_data']
val_data = np.moveaxis(val_data, 0, -1)
val_mask = val_data[:, :, -1]
val_data = val_data[:, :, :6]
val_labels = labelset['relabeled_val']



val_data_chips =   make_chips_data(val_data, chip_width, chip_height)
val_labels_chips = make_chips_labels(val_labels, chip_width, chip_height)
numChips = val_labels_chips.shape[0]
print("Total Number of Chips Taken from Mask: ", numChips)

print("val image shape: ", val_data.shape )

#hyperparameters
regularizer_coeff = 0.1

#Input layer for all layers and lambda split 
inputsX =  Input(shape = (40, 40, 6), batch_size = None) 

outputDenseLayers = []
importance_weights = []

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

##############################################################################################################

model = tf.keras.Model(inputs = [inputsX], outputs = output)
#Create summary of our model
model.summary()
#Compile the model
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.add_loss(Band_Importance(band_weights))
#######################################################################

#Load weights from train_#epochs_LR_L1Coeff.ckpt
trn_weights = "train_150_0001_baseline.ckpt"
model.load_weights(trn_weights)

# Update status of running program
print("Status: Program is running model. Please wait...")


def Band_Importance(band_weights):
    def _custom_loss():
        #abs value was to minimize more aggressively
        loss = abs(tf.norm(band_weights[0])) + abs(tf.norm(band_weights[1])) + abs(tf.norm(band_weights[2])) + abs(tf.norm(band_weights[3])) + abs(tf.norm(band_weights[4])) + abs(tf.norm(band_weights[5]))
        return loss
    return _custom_loss



# Predict for test data 
print("Chip size: ", val_data_chips.shape)
valPredict = model.predict(val_data_chips)
print("predict size: ", valPredict.shape)


# Calculate and display the error metrics, argmax to return class #
cMatrix = confusion_matrix(val_labels_chips.argmax(axis = 1), valPredict.argmax(axis = 1))
pScore = precision_score(val_labels_chips.argmax(axis = 1), valPredict.argmax(axis = 1), average = None)
rScore = recall_score(val_labels_chips.argmax(axis = 1), valPredict.argmax(axis = 1), average = None)

plt.clf()
plt.imshow(cMatrix, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Background','Trees', 'Grass']
plt.title('Confusion Matrix - Model Distinction of Trees vs Grass')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
for i in range(3):
    for j in range(3):
        plt.text(j,i, str(cMatrix[i][j]))
plt.show()

prTable = BeautifulTable()
prTable.columns.header = [" ", "Background", "Trees", "Grass"]
prTable.rows.append(["Precision", pScore[0], pScore[1], pScore[2]])
prTable.rows.append(["Recall", rScore[0], rScore[1], rScore[2]])
print(prTable)

new_label_image = make_image_from_chips(val_labels_chips, chip_width, chip_height)
new_label_image = np.array(new_label_image, dtype = int)

predict_image = make_image_from_chips(valPredict, chip_width, chip_height)
predict_image = np.array(predict_image, dtype = int)

#Background to black, trees to blue, grass to green
cmap = colors.ListedColormap(['k','b','g'])
plt.imshow(new_label_image, interpolation='nearest', cmap=cmap)
plt.title("Label Image")
plt.tight_layout()
plt.show()

#Background to black, trees to blue, grass to green
plt.imshow(predict_image, interpolation='nearest', cmap=cmap)
plt.title("Image Predictions by Model")
plt.tight_layout()
plt.show()
