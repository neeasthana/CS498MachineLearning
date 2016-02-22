##Neeraj Asthana (nasthan2)
##CS 498 HW3

#environment
options(warn=-1)
library(caret)
library(klaR)

same <- 1
different <- 0

#load data
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW3')
raw_train_data <- read.csv("pubfig_train_50000_pairs.txt", comment.char = "#", sep = "\t")

x_vector <- raw_train_data[,-1]
y_labels <- raw_train_data[,1]
face1 <- x_vector[,c(1,73)]
face2 <- x_vector[,c(74:146)]

#create training and testing sets
datasplit <- createDataPartition(y=y_labels, p=.8, list=FALSE)
trainx <- x_vector[datasplit,]
trainy <- y_labels[datasplit]
otherx <- x_vector[-datasplit]
othery <- y_labels[-datasplit]
next_datasplit <- createDataPartition(y=y_labels, p=.5, list=FALSE)
valx <- otherx[next_datasplit]
valy <- othery[next_datasplit]
testx <- otherx[-next_datasplit]
testx <- othery[-next_datasplit]

##Problem 1 part 1 - Linear SVM

##Problem 1 part 2 - Naive Bayes
tr <- trainControl(method='cv' , number=10)
model <- train (trainx , factor(trainy), 'nb' , trControl=tr)
