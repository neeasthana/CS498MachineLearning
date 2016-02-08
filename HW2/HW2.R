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
steplength_a <- 1
steplength_b <- 1

#read files and create data set
raw_train_data <- read.csv('adult.data', header=FALSE, na.strings = "?")
raw_test_data <- read.csv('adult.test', header=FALSE, na.strings = "?")
raw_data <- rbind(raw_train_data, raw_test_data, make.row.names=FALSE)

#Split continuous and label data
#continuous variables: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
x_vector <- raw_data[,c(1,3,5,11,12,13)]
y_labels <- raw_data[,15]

#Mentions of positive and negative examples for reference
neg_example <- y_labels[1]
neg_example2 <- y_labels[48842]
pos_example <- y_labels[8]
pos_example2 <- y_labels[48843]

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

#SVM Training
#loss function for SVM:
hinge_loss <- function(predicted, actual){
  return (max(0, 1 - (predicted * actual) ))
}

#evaluation for specific example x (6 items in vector) with parameters a and b
evaluate <- function(x, a, b){
  new_x <- as.numeric(as.matrix(x))
  return (t(a) %*% new_x + b) 
}

#Change y in dataset from <=50k and >50k to -1 and 1
converty <- function(y){
  if(y == neg_example | y == neg_example2){
    return (-1)
  }
  else if(y == pos_example | y == pos_example2){
    return (1)
  }
  else{
    return(NA)
  }
}

for (lambda in lambdas){
  #random initialization
  a <- runif(dim(x_vector)[2], min=-.001, max=.001)
  b <- runif(1, min=0, max=.01)
  
  #set out 50 examples for testing after every 30 steps
  ran_vals <- sample(1:dim(trainx)[1], 50)
  accuracy_data <- trainx[ran_vals, ]
  accuracy_labels <- trainy[ran_vals]
  train_data <- trainx[-ran_vals,]
  train_labels <- trainy[-ran_vals]
  
  for (epoch in 1:epochs){
    for (step in 1:steps){
      k <- sample(1:length(train_labels), 1)
      xex <- as.numeric(as.matrix( train_data[k,] ))
      yex <- converty( train_labels[k] )
      
      pred <- evaluate(xex, a, b)
      steplength = 1 / ((steplength_a * epoch) + steplength_b)
      
      #gradient vectors
      if(yex * pred >= 1){
        p1 <- lambda * a
        p2 <- 0
      }
      else {
        p1 <- (lambda * a) - (yex * xex)
        p2 <- -(yex)
      }
      
      #update values for a and b by gradient descent
      a <- a - (steplength * p1)
      b <- b - (steplength * p2)
    }
  }
}

##Problem 2.5a

##Problem 2.5b

##Problem 2.5c