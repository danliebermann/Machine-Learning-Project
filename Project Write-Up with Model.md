

Practical Machine Learning Course Project Submission
========================================================

*Executive Summary*
This analysis attempts to predict how well an individual did a certain activity (in this instance barbell lifts), based on motion data from accelerometers worn on belt, forearm, arm, and dumbbell.  In this instance participants were lifting barbells correctly and incorrectly in five different ways labeled A through E.  Using data from 19,622 instances of barbell lifts, we were able to develop a predictive model using the stochastic gradient boosting machine learning approach that was able to correctly predict the type of barbell lift being performed with an expected 99.7% out of sample accuracy.

*Analysis*
The analysis begins by downloading the caret package to perform predictive modeling and the doMC package to model using multiple cores.

```r
#Add packages to library
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

Data comes from the Groupware@LES website which can be found here: http://groupware.les.inf.puc-rio.br/har. The data has been pre-divided into training and test sets.  We download each csv individually and save it to a data frame.


```r
#Download training data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "./Training.csv", method = "curl")
training<-read.csv("Training.csv")
#Download testing data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "./Testing.csv", method = "curl")
testing<-read.csv("Testing.csv")
```

The training set is much larger than the test set as shown below.  The two sets contain identical columns, except the training set includes a variable called "classe" which identifies the type of activity being performed from A - E.  The testing set include a column for problem numbers to be solved.

```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

The dataset is then cleaned in two ways.  First, columns including only NA values are removed from both the training and testing sets.  Following this, both sets have identical columns with the exception of the classe/problem number columns


```r
#Clean the test data by selecting only columns with no NA values
test<-testing[, colSums(is.na(testing))==0]
#Select only columns from the training set that are in the test  set
train<-training[,colnames(training)%in%colnames(test)]
#Add the classe variable back into the training set
train<-cbind(train, training$classe)
colnames(train)[60]<-"classe"
```

Second, the column titled "X" is removed.  "X" captures the row number, and since the classe variable appears in order in the dataset, this row variable capture information that is likely not applicable out of sample.  For this reason, this column is removed.


```r
#Strip row number variable since it does not likely provide information applicable to the test set
train<-train[,-1]
test<-test[,-1]
```

The remaining variables relate only to information that should be applicable across both the training and the test sets regarding: the participant, the window, and the accelerometer measurements.  To begin, we set the seed to ensure reproducibility and register cores to machine learn in parallel.


```r
#Set the seed for reproducibility
set.seed(123)
#Register cores
registerDoMC(cores = 2)
```

From here we experimented with different machine learning approaches using caret's train function to determine which one provides the best out of sample accuracy.  This will not be shown here due to processing limitations, but approaches tested included predicting with trees, generalized linear models, and naive Bayes. In all instances, expected out of sample accuracy was lower than 55%.  Boosting is a popular machine learning method that takes an iterative approach to classifying observations with the objective of minimizing error.  In each iteration, misclassified observations are weighted based on their errors.

Repeated k-fold cross-validation was used to estimate the out of sample accuracy/error.  This method divides the training data into a number of folds, k, and then repeats this multiple times.  In this cases I used 10 folds and repeated this 10 times.


```r
##Determine cross-validation approach
fitControl<-trainControl(method = "repeatedcv", number = 10, repeats = 10)
#Predict classe based on training variables using stochastic gradient boosting
modelFit<-train(classe ~., data=train, trControl = fitControl, method="gbm", verbose =FALSE)
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loaded gbm 2.1
## Loading required package: plyr
```

```r
modelFit
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    58 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## 
## Summary of sample sizes: 17660, 17661, 17659, 17660, 17658, 17660, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy  Kappa  Accuracy SD  Kappa SD
##   1                   50      0.839     0.796  0.00980      0.01252 
##   1                  100      0.900     0.873  0.00708      0.00898 
##   1                  150      0.927     0.908  0.00645      0.00820 
##   2                   50      0.958     0.946  0.00529      0.00670 
##   2                  100      0.987     0.984  0.00234      0.00296 
##   2                  150      0.992     0.990  0.00179      0.00227 
##   3                   50      0.984     0.980  0.00315      0.00398 
##   3                  100      0.994     0.992  0.00167      0.00212 
##   3                  150      0.997     0.996  0.00115      0.00145 
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3 and shrinkage = 0.1.
```

Accuracy was used to select the optimal model. The expected out of sample error is 0.3%, based on the model's ability to accurately predict 99.7% of cases accuracy during cross-validation. 

The model's strong performance can further be visualized using the confusion matrix, which shows classifications by percent.


```r
#Print confusion matrix of outcomes
confusionMatrix(modelFit)
```

```
## Cross-Validated (10 fold, repeated 10 times) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.0  0.0  0.0  0.0
##          B  0.0 19.3  0.0  0.0  0.0
##          C  0.0  0.0 17.3  0.0  0.0
##          D  0.0  0.0  0.1 16.3  0.0
##          E  0.0  0.0  0.0  0.0 18.3
```

Finally, we predict answers for the test data using the model.  Following submission on Coursera, it was determined that all predictions were made accurately.

```r
#Predict and print the class of activity for the test set using the stochastic gradient boosting model
answers<-predict(modelFit, test)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

