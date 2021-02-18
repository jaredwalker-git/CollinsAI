import os
from osgeo import gdal
import pandas as pd
from scipy.io import loadmat
from pyrsgis.convert import changeDimension
import numpy as np

#print("Enter Directory:")
#userinput = input()
#chdir = os.chdir(userinput)

os.chdir('C:\\Users\\Jared\\Documents\\Datasets')
rit18data = loadmat('rit18_data.mat')


trainData = rit18data['train_data']
trainLabels = rit18data['train_labels']

valData = rit18data['val_data']
valLabels = rit18data['val_labels']


print("Multispectral image shape: ", trainData.shape)
print("Label array shape: ", trainLabels.shape)


print("Test data shape: ", valData.shape)
print("Test label shape: ", valLabels.shape)




#change to 1d array from numpy array where columns are bands and rows are pixels
trainData = changeDimension(trainData)
trainLabels = changeDimension(trainLabels)

nBands = trainData.shape[1]



print("New Feature image shape: ", trainData.shape)
print("New Label image shape: ", trainLabels.shape)




# Normalise the data
#trainData = trainData / 255.0
#xTest = xTest / 255.0
#featuresHyderabad = featuresHyderabad / 255.0

# Reshape the data
trainData = trainData.reshape((trainData.shape[0], 1, trainData.shape[1]))


# Print the shape of reshaped data
print(trainData.shape, trainLabels.shape)



from tf import keras

# Define the parameters of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(2, activation='softmax')])

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
model.fit(xTrain, yTrain, epochs=2)





from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Predict for test data 
yTestPredicted = model.predict(xTest)
#removes first column (inputs) of yTestPredicted
yTestPredicted = yTestPredicted[:,1]




# Calculate and display the error metrics
yTestPredicted = (yTestPredicted>0.5).astype(int)
cMatrix = confusion_matrix(yTest, yTestPredicted)
pScore = precision_score(yTest, yTestPredicted)
rScore = recall_score(yTest, yTestPredicted)

print("Confusion matrix: for 14 nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))







predicted = model.predict(featuresHyderabad)
predicted = predicted[:,1]

#Export raster to ds3 -> datasource for hyderabad
prediction = np.reshape(predicted, (ds3.RasterYSize, ds3.RasterXSize))
outFile = 'Hyderabad_2011_BuiltupNN_predicted.tif'
raster.export(prediction, ds3, filename=outFile, dtype='float')





























