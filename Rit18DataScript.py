# Pre-installs required.
'''
You need to install the correct pythons libraries as required from Windows'CMD:
pip install scipy
pip install pandas
...etc
'''

# import libaries
#from scipy.io import loadmat
import scipy.io.matlab as sio
import os
import numpy as np
import pandas as pd


# reading in data
os.chdir('C:\\Users\\Jared\\Documents\\GitHub\\CollinsAI') #Change folder_path based on your PC.
rit18 = sio.loadmat('rit18_asphalt_vegetation.mat') #file_name


print(rit18.keys())
'''
# returns dictionary index for
# comment/uncomment as needed for print commands

#print(type(rit18['train_data']) , rit18['train_data'].shape, rit18['train_labels'].shape, rit18['classes'])

#returns dictionary index
#print(rit18.keys())
#print(type(rit18['train_data']) , rit18['train_data'].shape, rit18['train_labels'].shape, rit18['classes'])

trainLabels = rit18['train_labels']
count = 0

#Cesar: Choose class that we wish to train our model with 
for pixel in trainLabels.flat:
    if pixel == 18:
        print(pixel)
        break
    elif pixel == 2:
        break
    elif pixel == 14:
        break
    else:
        pixel = 0
        count += count

# Temporary minimization of data until label normalization is done
print("done")
#Returns data shape

rit18['train_labels'] = trainLabels
scipy.io.savemat(rit18_data_3Class.mat)

#Creating Chunks - adds dimensionality - now (7, 3131, 403, 3, 14) - need to reference label data array - 2(Trees), 14(Grass), 18(Asphault)
goodDataArray = []'''

''' def find_good_datapoints():
    labelRows = 0
    goodPoints = 0
    
    for labelRows in range(9393): #Indexing through the array by each column of given row
    
        for labelColumns in range(5642):
         
            if trainLabels[labelRows, labelColumns] == 2 or trainLabels[labelRows, labelColumns] ==  14 or trainLabels[labelRows, labelColumns] ==  18 :
                goodDataArray.append([labelRows, labelColumns])
                goodPoints += 1
               
                
    print(goodPoints) 


find_good_datapoints()
'''

