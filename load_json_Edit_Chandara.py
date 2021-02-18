import json
import matplotlib.pyplot as plt
import numpy as np

import os

'''
DSTL Dataset | json of the polygon data instead of the wkt format.
Therefore, it is easier to work with the json data.
Program is used to help us get started on matching the labels with the images.
Purpose: Load multiple materials for each sub image.  
'''

##### = change as needed

##### Which image folder to open
select_imageFolderName = "6010_1_2" 

image_path = "train_geojson_v3" + '/' + select_imageFolderName + '/'
# Determines the number of loops for amount of materials on a sub image    
list = os.listdir(image_path)
number_files = len(list)
# subtract 1 because of Grid file
loop_NumMaterials = number_files - 1

# loop for each file_name 
RunEachMatFiles = 0

while loop_NumMaterials != 0:


    # download json folder and extract to same destination as this program
    folder_path = image_path
    file_name = os.listdir(folder_path)[RunEachMatFiles]
    print (file_name)

    # condition set for changing each file_name
    if RunEachMatFiles != number_files:
        RunEachMatFiles += 1

    f = open(folder_path + file_name) 

    # returns JSON object as a dictionary 
    data = json.load(f)

    # while loop condition for every materials
    loop_NumMaterials -= 1

    # see entire dictionary
    #print(data)

    # Returns a list of all files and folders in a directory
    # entries = os.listdir('train_geojson_v3/6010_1_2/')
    #print(entries)

    # dig through the dictionary layer by layer to find where the point data and material
    # currently printing only points, color coding does not coorespond to material
    for feature in data["features"]:
        polygon = np.array(feature["geometry"]["coordinates"][0])

        print(polygon)

        # scatter requires a list of points for each axis x and y
        # polygon is currently a list of points [x,y], [x,y]...
        # perform the transpose to get 2 lists, one for x, and one for y
        # [x0,x1,x2,x3...], [y0,y1,y2,y3...]
        x,y = polygon.T

        # perform scatter to add points, perform plot to connect points with lines
        # performing scatter as well for stylistic purposes
        plt.scatter(x,y)
        plt.plot(x, y)

'''
Decide on how to display Figure | (1 Materials per Figure) vs. (All Materials per Figure)
'''
    ##### (Only 1 Material per Figure) will display onto user PC's screen
    # uncomment and change if needed
        #plt.title(filename)
        #plt.show()
    
###### (All Materials per Figure) will display onto user PC's screen
# uncomment to see larger materials set, may cause lag to PC due to large datapoints.
plt.title('All materials included for sub image' + '(' + select_imageFolderName + ')')
plt.show()

    # use magnifying glass tool once image loads to zoom into polygons

