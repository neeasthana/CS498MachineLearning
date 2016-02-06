##Neeraj Asthana (nasthan2)
##CS 498 HW2 Problem 2.5

#environment
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW2')
library(caret)

#parameters
lambda <- c(.001, .01, .1, 1)
epochs <- 50
steps <- 300
percent_training <- .8
percent_validation <- .1
percent_test <- .1
num_examples_epoch_test <- 50
steps_til_eval <- 30

#read files and create data set
raw_train_data <- read.csv('adult.data', header=FALSE, na.strings = "?")
raw_test_data <- read.csv('adult.test', header=FALSE, na.strings = "?")
raw_data <- rbind(raw_train_data, raw_test_data)

##Problem 2.5a
#continuous variables: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week

##Problem 2.5b

##Problem 2.5c