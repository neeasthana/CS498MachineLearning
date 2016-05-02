##Neeraj Asthana (nasthan2)
##CS 498 HW Extra Credit

library(glmnet)

#setup environment
setwd("/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HWEC")

#read input data
raw_data <- read.csv("genes.txt", header = FALSE, sep = " ")
data <- t(as.matrix(raw_data))

#read labels
raw_tumors <- read.csv("tumors.txt", header = FALSE, sep = " ")
tumors <- c(raw_tumors < 0)*1

#preform logistic regression using cross validated lasso logistic regression
set.seed(1)
model <- cv.glmnet(data, tumors, family = "binomial", type.measure = "deviance", alpha = 1, nfolds = 6)

#evaluate performance of model (deviance)
plot(model)
minlambda <- model$lambda.min
deviance <- model$cvm[which(model$lambda == minlambda)]

set.seed(1)
modelauc <- cv.glmnet(data, tumors, family = "binomial", type.measure = "auc", alpha = 1, nfolds = 6)
minauclambda <- modelauc$lambda.min
#get auc of model with the lowest deviance
auc <- modelauc$cvm[which(model$lambda == minlambda)]

minlambda
deviance
auc
