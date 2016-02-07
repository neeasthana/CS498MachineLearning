##Neeraj Asthana (nasthan2)
##CS 498 HW2 Problem 2.5

#environment
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW2')
library(caret)

#parameters
lambdas <- c(.001, .01, .1, 1)
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

#Split continuous and label data
#continuous variables: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
x_vector <- raw_data[,c(1,3,5,11,12,13)]
y_labels <- raw_data[,15]

#Mentions of positive and negative examples for reference
neg_example <- raw_test_data[2,15]
pos_example <- raw_test_data[4,15]

#split data into training, test, and validation sets
datasplit <- createDataPartition(y=y_labels, p=.8, list=FALSE)
trainx <- x_vector[datasplit,]
trainy <- y_labels[datasplit]
otherx <- x_vector[-datasplit,]
othery <- y_labels[-datasplit]
datasplit2 <- createDataPartition(y=othery, p=.5, list=FALSE)
testx <- otherx[datasplit2,]
testy <- othery[datasplit2]
valx <- otherx[-datasplit2,]
valy <- othery[-datasplit2]

##Problem 2.5a
hinge_loss <- function(predicted, actual){
  return (max(0, 1 - (predicted * actual) ))
}

step <- function(lambda){
  k <- sample(1:length(trainx), 1)
  xex <- trainx[k]
  yex <- trainy[k]
}

for (i in lambdas){
  print(i)
}

##Problem 2.5b

##Problem 2.5c