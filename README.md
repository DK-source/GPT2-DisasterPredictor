#### Disaster Predictor using GPT2

**Name:** Duy Pham  
**Email:** dkp3339@mavs.uta.edu  
**Department:** [Department of Data Science](https://www.uta.edu/academics/schools-colleges/science/departments/data-science)  
**University:** [The University of Texas at Arlington](https://www.uta.edu/)  
**Level:** Undergraduate - Sophomore (2nd year)  
**Position Title:** Student  

#### Description of the project's content  

This repository contains the following:

* **[Training](https://github.com/DK-source/DMC2021F/tree/main/Training):**  
    This directory contains the training code and the necessary assets to train the model,  
	assets includes the multiple datasets for training and testing.  
    <br>
* **[Output](https://github.com/DK-source/DMC2021F/tree/main/Output)**  
    This directory contains a copy of your dataset created from 'Predict.py',  
    	the dataset in .csv file now has a target column with an integer of 0 or 1.  
	See **Instruction to use** below for explanation.  
    <br>
* **Predict.py:**  
    This code is for others to use to predict the disaster datasets.  
    <br>

#### Instruction to use  
 
1. Due to github not allowing big files to be pushed to their server,  
you can either download a premade model [here](https://drive.google.com/drive/folders/1k7YH7RbHzAaQs7a8_EKfO-Y0qsJ0tAja?usp=sharing)  
or train the model yourselves before 'Predict.py' can be used.  
To do so, simply run 'The Model.py',  
inside the [Training](https://github.com/DK-source/DMC2021F/tree/main/Training) directory.  
The training takes approximately 3-4 hours, but oculd be lowered by changing the epochs within the code.  
2. After the model has downloaded or finished training,  
move the downloaded model to the [Training](https://github.com/DK-source/DMC2021F/tree/main/Training) folder if you havent,  
run 'Predict.py',  
and enter the path to the dataset to be predicted.  
(i.e., C:/Download/RandomTweets.csv)
3. The code will return a .csv files inside the [Output](https://github.com/DK-source/DMC2021F/tree/main/Output) folder.  
This file has an extra column containing an integer of 0 or 1.  
A number 0 mean the tweet is very likely not about a disaster occuring.  
Vice versa for a number 1.  
<br>
For troubleshooting, please contact:  

Duy Pham  
dkp3339@mavs.uta.edu  
