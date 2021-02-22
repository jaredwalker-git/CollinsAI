# Collins AI
Material Band Model



To Do: 

Research and adjust batch size and learning rate.
- lower train time and increase accuracy

Create Code Block to run model on one material vs. background.
- Our model ultimately will train distinguishing one material from background in order to determine bands used most for that material

Research and Implement Keras Loss function into learning model.
-Loss function that penalizes model for use of additional data  

Create Code Block to normalize Label Data Distribution.
-need data set with uniform distribution of labels

Research Logistic Regression/Feature Importance and Determine usability in initializing linear correlation of spectral response.
-peter mentioned we may be able to initialize a linear relationship between the spectral response of a material and band definition, this may be worth looking into



# Jared ->
2/18/21
-Concerns with DSTL - 
Datatype for model.fit() with json files

-First Successful Rit18 Run with decreasing accuracy. Learning rate needs to be adjusted

2/12/21
-Begining attempting to run the RIT18 data to the model, created a copy of the model to do so for the time being. Ideally the model will be able to accept multiple datasets but i currently need to understand more of the consderations involved before being able to take that step, for now i simply created a copy of the model to use for the RIT18 data.

2/11/21
-Block to input current working directory and input files was created for ease of use.
-Currently need to figure validity of DSTL as input and attempted to input DSTL but after meeting with peter it seems focus should be on data manipulation
-Rit18 data ready as backup and I will continue to work
-Added Directory call for user input

# Mika -> 


# Armando ->


# Cesar -> 



# Chandara
