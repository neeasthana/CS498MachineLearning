##Neeraj Asthana (nasthan2)
##CS 498 HW6

#references:
#http://www4.stat.ncsu.edu/~post/josh/LASSO_Ridge_Elastic_Net_-_Examples.html
#https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
#https://rpubs.com/chihst01/15922

library(glmnet)

##Problem 1 - linear regression

#setup
setwd("~/Documents/UIUC/CS 498/CS498MachineLearning/HW6/")

#read in csv file
raw_data <- read.csv("creditcard.csv", skip=1, header = TRUE)
x <- as.matrix(raw_data[,seq(2,24)])
y <- as.factor(raw_data[,25])




##Problem 2 - logistic regression

#setup
setwd("~/Documents/UIUC/CS 498/CS498MachineLearning/HW6")

#read in csv file
raw_data <- read.csv("creditcard.csv", skip=1, header = TRUE)
x <- as.matrix(raw_data[,seq(2,24)])
y <- as.factor(raw_data[,25])

#split into training and test data

#train models
res0 <- cv.glmnet(x,y, alpha=0, type.measure = "class",family='binomial') #ridge 
res2 <- cv.glmnet(x,y, alpha=.2, family='binomial')
res4 <- cv.glmnet(x,y, alpha=.4, family='binomial')
res5 <- cv.glmnet(x,y, alpha=.5, family='binomial')
res6 <- cv.glmnet(x,y, alpha=.6, family='binomial')
res8 <- cv.glmnet(x,y, alpha=.8, family='binomial')
res1 <- cv.glmnet(x,y, alpha=1, family='binomial') #lasso

#predictions

#see how many are misclassified