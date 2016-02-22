##Neeraj Asthana (nasthan2)
##CS 498 HW3

#environment
options(warn=-1)
library(caret)
library(klaR)
library(randomForest)

#load data
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW3')
raw_train_data <- read.csv("pubfig_train_50000_pairs.txt", comment.char = "#", sep = "\t")

x_vector <- raw_train_data[,-1]
y_labels <- raw_train_data[,1]
face1 <- x_vector[,c(1,73)]
face2 <- x_vector[,c(74:146)]



##Problem 1 part 1 - Linear SVM
#data held out for testing
datasplit <- createDataPartition(y=y_labels, p=.1, list=FALSE)
testx <- x_vector[datasplit,]
testy <- y_labels[datasplit]
otherx <- x_vector[-datasplit,]
othery <- y_labels[-datasplit]

lambdas <- c(.0001, .001,.01,.1,1)
val_accuracies <- c()
train_accuracies <- c()
test_accuracies <- c()
for (lambda in lambdas){
  #create training and testing sets
  next_datasplit <- createDataPartition(y=othery, p=(8/9), list=FALSE)
  trainx <- otherx[next_datasplit,]
  trainy <- othery[next_datasplit]
  valx <- otherx[-next_datasplit,]
  valy <- othery[-next_datasplit]
  
  #train SVM
  op <- paste("-c", sprintf('%f', lambda), sep=" ")
  svm <- svmlight(trainx, trainy, pathsvm="/home/neeraj/Documents/UIUC/svm_light", svm.options=op)
  
  #predict values from SVM model
  trainpred <- predict(svm, trainx)$class
  valpred <- predict(svm, valx)$class
  testpred <- predict(svm, testx)$class
  
  #accuracy calculations
  train_accuracies = c(train_accuracies, sum(trainpred==trainy)/length(trainy))
  val_accuracies = c(val_accuracies, sum(valpred==valy)/length(valy))
  test_accuracies = c(test_accuracies, sum(testpred==testy)/length(testy))
}



##Problem 1 part 2 - Naive Bayes
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
  ptrlogs<--(1/2)*rowSums(apply(ptrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  ntroffsets<-t(t(trainx)-ntrmean)
  ntrscales<-t(t(ntroffsets)/ntrsd)
  ntrlogs<--(1/2)*rowSums(apply(ntrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  lvwtr<-ptrlogs>ntrlogs
  
  pteoffsets<-t(t(testx)-ptrmean)
  ptescales<-t(t(pteoffsets)/ptrsd)
  ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  nteoffsets<-t(t(testx)-ntrmean)
  ntescales<-t(t(nteoffsets)/ntrsd)
  ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  lvwte<-ptelogs>ntelogs
  
  gotright_train <- lvwtr==trainy
  gotright_test <- lvwte==testy
  trscore[wi]<-sum(gotright_train)/(sum(gotright_train)+sum(!gotrighttr))
  tescore[wi]<-sum(gotright_test)/(sum(gotright_test)+sum(!gotright_test))
}
accuracy_train <- sum(trscore) / length(trscore)
accuracy_test <- sum(tescore) / length(tescore)


##Problem 1 part 3 - Random Forest
datasplit <- createDataPartition(y=y_labels, p=.9, list=FALSE)
trainx <- x_vector[datasplit,]
trainy <- y_labels[datasplit]
testx <- x_vector[-datasplit,]
testy <- y_labels[-datasplit]

rf <- randomForest(trainx, trainy)

trainpred <- predict(rf, trainx)
testpred <- predict(rf, testx)
train_accuracy <- sum((trainpred >= .5) == trainy)/length(trainy)
test_accuracy <- sum((testpred >= .5) == testy)/length(testy)
#train accuracy: 1, test accuracy: 0.8063613