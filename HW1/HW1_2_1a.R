##Neeraj Asthana (nasthan2)
##CS 498 HW1
options(warn=-1)
library(caret)
library(klaR)

##2.1a
#load data
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW1')
raw_data<-read.csv('pima.csv', header=FALSE)
x_vector <- raw_data[-c(9)]
y_labels <- raw_data[,9]

#train naive bayes model
trscore<-array(dim=10)
tescore<-array(dim=10)
for (wi in 1:10){
  #create training and testing sets
  datasplit <- createDataPartition(y=y_labels, p=.8, list=FALSE)
  trainx <- x_vector[datasplit,]
  trainy <- y_labels[datasplit]
  testx <- x_vector[-datasplit,]
  testy <- y_labels[-datasplit]
  
  #splitting positive and negative examples
  trposflag<-trainy>0
  positive_examples <- trainx[trposflag, ]
  negative_examples <- trainx[!trposflag,]
  
  #calculate means and sds
  ptrmean<-sapply(positive_examples, mean, na.rm=TRUE)
  ntrmean<-sapply(negative_examples, mean, na.rm=TRUE)
  ptrsd<-sapply(positive_examples, sd, na.rm=TRUE)
  ntrsd<-sapply(negative_examples, sd, na.rm=TRUE)
  
  #calculate offsets and scales
  ptroffsets<-t(t(trainx)-ptrmean)
  ptrscales<-t(t(ptroffsets)/ptrsd)
  
  pteoffsets<-t(t(testx)-ptrmean)
  ptescales<-t(t(pteoffsets)/ptrsd)
  ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  nteoffsets<-t(t(testx)-ntrmean)
  ntescales<-t(t(nteoffsets)/ntrsd)
  ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  lvwte<-ptelogs>ntelogs
  gotright<-lvwte==testy
  tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))
}
accuracy <- sum(tescore) / length(tescore)
#accuracy after cross validating 10 times
accuracy




##2.1b
#replace '0' in columns 
x_vector_copy <- x_vector
for (i in c(3, 4, 6, 8)){
  non_values <- x_vector[, i]==0
  x_vector_copy[non_values, i]=NA
}

#train naive bayes model
trscore<-array(dim=10)
tescore<-array(dim=10)
for (wi in 1:10){
  #create training and testing sets
  datasplit <- createDataPartition(y=y_labels, p=.8, list=FALSE)
  trainx <- x_vector_copy[datasplit,]
  trainy <- y_labels[datasplit]
  testx <- x_vector_copy[-datasplit,]
  testy <- y_labels[-datasplit]
  
  #splitting positive and negative examples
  trposflag<-trainy>0
  positive_examples <- trainx[trposflag, ]
  negative_examples <- trainx[!trposflag,]
  
  #calculate means and sds
  ptrmean<-sapply(positive_examples, mean, na.rm=TRUE)
  ntrmean<-sapply(negative_examples, mean, na.rm=TRUE)
  ptrsd<-sapply(positive_examples, sd, na.rm=TRUE)
  ntrsd<-sapply(negative_examples, sd, na.rm=TRUE)
  
  #calculate offsets and scales
  ptroffsets<-t(t(trainx)-ptrmean)
  ptrscales<-t(t(ptroffsets)/ptrsd)
  
  pteoffsets<-t(t(testx)-ptrmean)
  ptescales<-t(t(pteoffsets)/ptrsd)
  ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  nteoffsets<-t(t(testx)-ntrmean)
  ntescales<-t(t(nteoffsets)/ntrsd)
  ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  lvwte<-ptelogs>ntelogs
  gotright<-lvwte==testy
  tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))
}
accuracy <- sum(tescore) / length(tescore)
#accuracy after cross validating 10 times
accuracy




##2.1c
#create training and testing sets
test_accuracies <- array(dim=10)
for (wi in 1:10){
  datasplit <- createDataPartition(y=y_labels, p=.8, list=FALSE)
  trainx <- x_vector[datasplit,]
  trainy <- y_labels[datasplit]
  testx <- x_vector[-datasplit,]
  testy <- y_labels[-datasplit]

  #train naive bayes model using klaR package
  tr <- trainControl(method='cv' , number=10)
  model <- train (trainx , factor(trainy) , 'nb' , trControl=tr)

  #prediction
  predictions <- predict(model, newdata=testx)
  correct <- length(testy[testy == predictions])
  wrong <- length(testy[testy != predictions])
  accuracy <- correct / (correct + wrong)
  test_accuracies[wi] <- accuracy
}
cross_validation_accuracy <- sum(test_accuracies)/length(test_accuracies)
#accuracy after cross validating 10 times
cross_validation_accuracy




##2.1d
#create data sets for training and testing
datasplit<-createDataPartition(y=y_labels, p=.8, list=FALSE)
trainx <- x_vector[datasplit,]
trainy <- y_labels[datasplit]
testx <- x_vector[-datasplit,]
testy <- y_labels[-datasplit]

#train svm
svm <- svmlight(trainx, factor(trainy), pathsvm="/home/neeraj/Documents/UIUC/svm_light")
labels <- predict(svm, testx)
answers <- labels$class

#accuracy
correct <- sum(answers == testy)
wrong <- sum(answers != testy)
accuracy <- correct / (correct + wrong)
accuracy