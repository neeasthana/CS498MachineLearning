##Neeraj Asthana (nasthan2)
##CS 498 HW3

#environment
options(warn=-1)
library(caret)
library(klaR)
library(randomForest)
library(class)
library(neuralnet)

#load data
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW3')
raw_train_data <- read.csv("pubfig_train_50000_pairs.txt", comment.char = "#", sep = "\t")
val1_data <- read.csv("pubfig_kaggle_1.txt", comment.char = "#", sep="\t")
val1_sol <- read.csv("pubfig_kaggle_1_solution.txt", header=TRUE)[-1,2]
val2_data <- read.csv("pubfig_kaggle_2.txt", comment.char = "#", sep="\t")
val2_sol <- read.csv("pubfig_kaggle_2_solution.txt", header=TRUE)[-1,2]
val3_data <- read.csv("pubfig_kaggle_3.txt", comment.char = "#", sep="\t")
val3_sol <- read.csv("pubfig_kaggle_3_solution.txt", header=TRUE)[-1,2]
test_data <- read.csv("pubfig_kaggle_eval.txt", comment.char = "#", sep = "\t")

x_vector <- raw_train_data[,-1]
y_labels <- raw_train_data[,1]
face1 <- x_vector[,c(1,73)]
face2 <- x_vector[,c(74:146)]


#scaling (prepare for easier neural networks)
scaled_raw <- scale(raw_train_data)
scaled_raw[,1] <- y_labels
names <- c("label")
for (i in 1:146){
  names <- c(names, paste("feature",i, sep=""))
}
colnames(scaled_raw) = names


##Problem 1 part 1 - Linear SVM
#data held out for testing
datasplit <- createDataPartition(y=y_labels, p=.1, list=FALSE)
valx <- x_vector[datasplit,]
valy <- y_labels[datasplit]
trainx <- x_vector[-datasplit,]
trainy <- y_labels[-datasplit]

lambdas <- c(.001,.01,.1,1)
val_accuracies <- c()
train_accuracies <- c()
full_accuracies <- c()
val1_accuracies <- c()
val2_accuracies <- c()
val3_accuracies <- c()
for (lambda in lambdas){
  #train SVM
  op <- paste("-c", sprintf('%f', lambda), sep=" ")
  svm <- svmlight(trainx, trainy, pathsvm="/home/neeraj/Documents/UIUC/svm_light", svm.options=op)
  
  #predict values from SVM model
  trainpred <- predict(svm, trainx)$class
  fulltrainpred <- predict(svm, x_vector)$class
  valpred <- predict(svm, valx)$class
  val1pred <- predict(svm, val1_data)$class
  val2pred <- predict(svm, val2_data)$class
  val3pred <- predict(svm, val3_data)$class
  
  #accuracy calculations
  train_accuracies = c(train_accuracies, sum(trainpred==trainy)/length(trainy))
  full_accuracies = c(full_accuracies, sum(fulltrainpred==y_labels)/length(y_labels))
  val_accuracies = c(val_accuracies, sum(valpred==valy)/length(valy))
  val1_accuracies = c(val1_accuracies, sum(val1pred==val1_sol)/length(val1_sol))
  val2_accuracies = c(val2_accuracies, sum(val2pred==val2_sol)/length(val2_sol))
  val3_accuracies = c(val3_accuracies, sum(val3pred==val3_sol)/length(val3_sol))
}



##Problem 1 part 2 - Naive Bayes
trscore<-array(dim=10)
val1score<-array(dim=10)
val2score<-array(dim=10)
val3score<-array(dim=10)
for (wi in 1:10){
  #create training and testing sets
  datasplit <- createDataPartition(y=y_labels, p=.8, list=FALSE)
  trainx <- x_vector
  trainy <- y_labels
  
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
  
  #testing predictions
  pteoffsets<-t(t(val1_data)-ptrmean)
  ptescales<-t(t(pteoffsets)/ptrsd)
  ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  nteoffsets<-t(t(val1_data)-ntrmean)
  ntescales<-t(t(nteoffsets)/ntrsd)
  ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  lvwte1<-ptelogs>ntelogs
  
  pteoffsets<-t(t(val2_data)-ptrmean)
  ptescales<-t(t(pteoffsets)/ptrsd)
  ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  nteoffsets<-t(t(val2_data)-ntrmean)
  ntescales<-t(t(nteoffsets)/ntrsd)
  ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  lvwte2<-ptelogs>ntelogs
  
  pteoffsets<-t(t(val3_data)-ptrmean)
  ptescales<-t(t(pteoffsets)/ptrsd)
  ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  nteoffsets<-t(t(val3_data)-ntrmean)
  ntescales<-t(t(nteoffsets)/ntrsd)
  ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  lvwte3<-ptelogs>ntelogs
  
  gotright_train <- lvwtr==trainy
  gotright_val1 <- lvwte1==val1_sol
  gotright_val2 <- lvwte2==val2_sol
  gotright_val3 <- lvwte3==val3_sol
  trscore[wi]<-sum(gotright_train)/(sum(gotright_train)+sum(!gotright_train))
  val1score[wi]<-sum(gotright_val1)/(sum(gotright_val1)+sum(!gotright_val1))
  val2score[wi]<-sum(gotright_val2)/(sum(gotright_val2)+sum(!gotright_val2))
  val3score[wi]<-sum(gotright_val3)/(sum(gotright_val3)+sum(!gotright_val3))
}
accuracy_train <- mean(trscore)
accuracy_val1 <- mean(val1score)
accuracy_val2 <- mean(val2score)
accuracy_val3 <- mean(val3score)


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

##Problem 2


##Problem 3
n <- names
f <- as.formula(paste("label ~", paste(n[!n %in% "label"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(40,10),linear.output=T)