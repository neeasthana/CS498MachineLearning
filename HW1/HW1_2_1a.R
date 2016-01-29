library(caret)

setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW1')
raw_data<-read.csv('pima.csv', header=FALSE)

x_vector <- raw_data[-c(9)]
y_labels <- raw_data[9]

#create training and testing sets
datasplit <- createDataPartition(y=y_labels, p=.8, list=False)
trainx <- xvector[datasplit,]
trainy <- yvector[datasplit]
testx <- xvector[-datasplit]
testy <- yvector[-datasplit]