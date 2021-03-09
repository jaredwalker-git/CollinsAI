# Collins AI
Material Band Model


# Jared ->
3/9/21 
-Among a bunch of smaller items, created loop to determine good datapoints which equate to about half the dataset. Seems that breaking the dataset into spatial pieces will be best route for training
-Also should be noted that the input data shape will now no longer be flattened per alex. Data shape change has therefor been removed

3/2/21
-Created seperate training and validation scripts so that validation tests can be run. idea is that user can run training script until they get the weights how they want em then save em and not have to retrain, then just run validation/ testing. Maybe we have validation script ask if one wants to validate or test using if statement? we will have to talk more on voice just wanted to let yall know
also this way we can easily run validation tests on our model when the time comes
i kept it seperate from the original file as to not confuse anyone too much but additional work should probably be done to these seperate files

2/23/21 
-Changed batch size and minimized training data to 500,000 units for now to lower training time
-This minimization of data will be directly implemented when label distribution is made uniform 

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
