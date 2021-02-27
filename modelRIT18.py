#Code is running on Python 3.7.9 version

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

# NOTE: Alter filepath + name lines below to personal specified file address
filepath = "C:\\Users\\Ahboy\\Desktop\\Datasets"
filename = "rit18_data.mat"
os.chdir(filepath)


# Update status of running program
print("Status: Program is running correctly. (ignore-Warning Signs)")

# Loads all variables stored in the MAT-file into a simple Python data structure
rit18data = loadmat(filename)

trainData = rit18data['train_data']
trainLabels = rit18data['train_labels']

valData = rit18data['val_data']
valLabels = rit18data['val_labels']

# Reshape the array in-place by assigning a tuple of array dimensions to it
print("Multispectral image shape: ", trainData.shape)
print("Label array shape: ", trainLabels.shape)

print("Test data shape: ", valData.shape)
print("Test label shape: ", valLabels.shape)


# Change to 1d array from numpy array where columns are bands and rows are pixels
trainData = changeDimension(trainData)
trainLabels = changeDimension(trainLabels)

valData = changeDimension(valData)
valLabels = changeDimension(valLabels)


nBands = trainData.shape[1]


print("New Feature image shape: ", trainData.shape)
print("New Label image shape: ", trainLabels.shape)



# Normalise the data
#trainData = trainData / 255.0
#xTest = xTest / 255.0
#featuresHyderabad = featuresHyderabad / 255.0

# Reshape the data to fit format of flattened input layer
trainData = trainData.reshape((trainData.shape[0], 1, trainData.shape[1]))
valData = valData.reshape((valData.shape[0], 1, valData.shape[1]))

# Temporary minimization of data until label normalization is done
chooseData = np.random.randint(52995306, size = 500000) #500k out of 52M random inputs is chosen

trainData = trainData[chooseData]
trainLabels = trainLabels[chooseData]

valData = valData[chooseData]
valLabels = valLabels[chooseData]
trainData.shape

# Print the shape of reshaped data
print(trainData.shape, trainLabels.shape)



from tensorflow import keras

# Define the parameters of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(36, activation='relu'),
    keras.layers.Dense(19, activation='softmax')])

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Update status of running program
print("Status: Program is training model. Please wait...")

# Run the model
model.fit(trainData, trainLabels, epochs=10, batch_size = 64)



from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Predict for test data 
valPredict = model.predict(valData)
# removes first column (inputs) of valPredict
valPredict = valPredict[:,1]




# Calculate and display the error metrics
valPredict = (valPredict>0.5).astype(int)
cMatrix = confusion_matrix(valLabels, valPredict)
pScore = precision_score(valLabels, valPredict, average = None)
rScore = recall_score(valLabels, valPredict)

print("Confusion matrix: for nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))


