##Neeraj Asthana (nasthan2)
##CS 498 HW1

library(caret)
library(klaR)

##2.1a
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



##2.1b
#replace '0' in columns 
x_vector_copy <- x_vector
for (i in c(3, 4, 6, 8)){
  non_values <- x_vector[, i]==0
  x_vector_copy[non_values, i]=NA
}

