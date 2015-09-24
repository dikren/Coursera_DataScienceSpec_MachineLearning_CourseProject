
# Prediction of quality of physical exercise performance.
## Executive summary.
The aim of this analysis is to train a machine learning model to determine if a person doing exercise in a wrong way.  
During the work an exploratory data analysis were performed for reducing the number of variables; 
the random forest model were trained and the estimation of the out of sample error were calculated. 
Also, the prediction for the test data were performed.

## Exploratory analysis and data cleaning.

First of all I need to load the data from the working directory.

```r
originalTrainData = read.csv("data/pml-training.csv", header = TRUE)
originalTestData = read.csv("data/pml-testing.csv", header = TRUE)
```

Apparently, there are very much columns with almost only empty or NA values:

```r
options(width = 100L)
originalTrainData[1:6, 10:25]
```

```
##   yaw_belt total_accel_belt kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 1    -94.4                3                                                         
## 2    -94.4                3                                                         
## 3    -94.4                3                                                         
## 4    -94.4                3                                                         
## 5    -94.4                3                                                         
## 6    -94.4                3                                                         
##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt max_picth_belt
## 1                                                                      NA             NA
## 2                                                                      NA             NA
## 3                                                                      NA             NA
## 4                                                                      NA             NA
## 5                                                                      NA             NA
## 6                                                                      NA             NA
##   max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt amplitude_roll_belt amplitude_pitch_belt
## 1                         NA             NA                               NA                   NA
## 2                         NA             NA                               NA                   NA
## 3                         NA             NA                               NA                   NA
## 4                         NA             NA                               NA                   NA
## 5                         NA             NA                               NA                   NA
## 6                         NA             NA                               NA                   NA
```

All those columns have some summary data for the time window of one exercise performed by a person in observation.
I am going to try to make a prediction model basing on the raw data coming from the devices without any statistics 
which can be calculated only after a while since the exercise was started.  
So, I will drop these columns. 


```r
columnsOfInterest = setdiff(names(originalTrainData), 
    grep("^var_|^stddev_|^min_|^max_|^skewness_|^avg_|^kurtosis_|^amplitude_|^var_", 
    names(originalTrainData), value = T))
```

Also there are indexing columns and columns with time and marker columns, 
which allowes to determine the number of time window and if current row is the end of the window 
and so the statistics summary can be calculated. All these columns are also redundant.  

I am interested only in:  
- the name of a person, since different persons may perform same task in different ways  
- the raw data coming from devices  
- classe column, since it is the output
- problem_id - it is the number of test task from the submission assignment in the test dataset.

So I am going to drop all other columns

```r
columnsOfInterest = setdiff(columnsOfInterest, 
    grep("^X$|raw_timestamp_part_1|raw_timestamp_part_2|cvtd_timestamp|new_window|num_window", 
    columnsOfInterest, value = T))

# I also remove "classe" column to get vector of columns common for the training and testing datasets.
# I will add classe and problem_id columns to each of them when it will be needed
columnsOfInterest = setdiff(columnsOfInterest, "classe")
```
Here are the columns I will use:

```r
columnsOfInterest
```

```
##  [1] "user_name"            "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [5] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
##  [9] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [17] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [21] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [29] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [33] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [41] "roll_forearm"         "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [45] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [53] "magnet_forearm_z"
```

Now I will make one dataset which I will use to train model and to estimate out of sample error, 
and another which I will need to perform assignment submission.

```r
sampleData = subset(originalTrainData, select = c(columnsOfInterest, "classe"))
submissionTestData = subset(originalTestData, select = c(columnsOfInterest, "problem_id"))
```

## Training of statistical model and estimation of out of sample error. 

I decided to use random forest algorithm for this analysis because of its good performance in classification tasks.
For the purpose of out of sample error estimation I used cross validation by specifying parameters for the train 
function, namely, I called trainControl function with method "cv" which stands for cross validation and with
default number of folds (= 10).  
So the random forest algorithm will be applied 10 times each time to only 9 of 10 folds (parts) of training dataset. 
Then the error on the left part will be calculated (for each of 10 left parts). 
The resulted out of sample error estimation is the average of these errors on all left parts.



```r
# loading caret package
suppressMessages(library(caret))
# making trainControl object with cv parameter specified (stands for cross validation). 
# The default k = 10 folds is used.
trainCtrl = trainControl(method = "cv")
# training random forest with cross validation for the purpose of estimation of out of sample error value
modelRF = train(classe ~ ., sampleData, model = "rf", trControl = trainCtrl)
```

Here is the results

```r
modelRF
```

```
## Random Forest 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17660, 17659, 17659, 17659, 17662, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9953112  0.9940688  0.001397837  0.001768573
##   29    0.9955150  0.9943266  0.001397523  0.001768012
##   57    0.9900620  0.9874274  0.002834749  0.003587661
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```
So the estimation of out of sample error is 1 - Accuracy = 1 - 
0.995515 = 
0.004485.  
And this is very good results I suppose.  
The per class correctness level can be seen below:

```r
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, model = "rf") 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 29
## 
##         OOB estimate of  error rate: 0.42%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5576    2    1    0    1 0.0007168459
## B   20 3774    3    0    0 0.0060574137
## C    0    9 3404    9    0 0.0052600818
## D    0    0   22 3191    3 0.0077736318
## E    0    1    5    6 3595 0.0033268644
```


## Predicting outcome for submission testing dataset. 

The testing data set must be exposed to the same data transformations as the training one. The only difference is
if I had been used some statistics of the training sample (function of the training data points) then for a test sample
I should to use the same value of statistics I have got on the training data.  
But in the current case I didn't use any statistic of the training data. I just decreased number of columns.  
So, below is the prediction of classe variable which was made by my trained model of random forest.

```r
# prediction of classe for the 20 size test data.
classe = predict(modelRF, newdata = subset(submissionTestData, select = -problem_id))
answers = cbind(submissionTestData, classe)
```

And here is the code for saving results into files.

```r
pml_write_files = function(dataFrame){
    n = dim(dataFrame)[1]
    res = character()
    for(i in 1:n){
        filename = paste0("testPredictions/problem_id_",i,".txt")
        classe = subset(dataFrame, subset = problem_id == i, select = classe)
        res = levels(classe[,1])[classe[1,1]]
        write.table(as.character(res),file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        
    }
}
pml_write_files(answers)
```

And the results are:

```r
res = answers$classe
names(res) = answers$problem_id
res
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
