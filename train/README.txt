This file contains instructions to train the model.

---- TODO ----
Please follow below step-by-step process to run the program.
Step 1 : Open Anaconda Prompt and go to the path of the folder train_program.py.
Step 2 : Run train_program.py program using below command in Anaconda Prompt:
python train_program.py ua.base ua.test

---- FILES ----
config.conf : This is a config file which contains informations for the training program.
ua.base : This is the training dataset- downloaded from MovieLens website
ua.test : This is the testing dataset - downloaded from the MovieLens website.

--- BELOW ARE INSTRUCTIONS IF USER WANTS TO CHANGE THE FUNCTIONALITY OF PROGRAM ---
1. Open config.conf file
    a. To change the number of item and user : Change num_item and num-user. Line 17 and 18. (Note: Different MovieLens Dataset have different number of item and users)
    b. To change the number training round : Change the num_round. Line 41. (Note: This generates n number of model files)
