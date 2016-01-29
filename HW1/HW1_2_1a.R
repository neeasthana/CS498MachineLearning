library(caret)
library(klaR)

#load data
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW1')
raw_data<-read.csv('pima.csv', header=FALSE)
x_vector <- raw_data[-c(9)]
y_labels <- raw_data[,9]

#create training and testing sets
datasplit <- createDataPartition(y=y_labels, p=.8, list=FALSE)
trainx <- x_vector[datasplit,]
trainy <- y_labels[datasplit]
testx <- x_vector[-datasplit,]
testy <- y_labels[-datasplit]

#train naive bayes model
tr <- trainControl(method='cv' , number=10)
model <- train (trainx , factor(trainy) , 'nb' , trControl=tr)

#prediction
predictions <- predict(model, newdata=testx)
confusionMatrix (data=predictions, testy)