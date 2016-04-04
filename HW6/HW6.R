##Neeraj Asthana (nasthan2)
##CS 498 HW6

#references:
#http://www4.stat.ncsu.edu/~post/josh/LASSO_Ridge_Elastic_Net_-_Examples.html
#https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
#https://rpubs.com/chihst01/15922

library(glmnet)
library(MASS)

###Problem 1 - linear regression

#setup
setwd("~/Documents/UIUC/CS 498/CS498MachineLearning/HW6/Geographical Original of Music")

#read in csv file and create features and predictors
raw_data <- read.csv("default_features_1059_tracks.txt", header = FALSE)
num_features <- dim(raw_data)[2]
num_examples <- dim(raw_data)[1]
x <- raw_data[,1:(num_features-2)]
#scaled by 90 to get box cox to work
latitude <- raw_data[,(num_features-1)] + 90
longitude <- raw_data[,num_features] + 90

##Simple linear regression
latfit <- lm(latitude ~ as.matrix(x))
summary(latfit)
latfitmse <- sum(latfit$residuals^2)/num_examples
latfitmse
par(mfrow=c(2,2))
plot(latfit)

longfit <- lm(longitude ~ as.matrix(x))
summary(longfit)
longfitmse <- sum(longfit$residuals^2)/num_examples
longfitmse
par(mfrow=c(2,2))
plot(longfit)


##Boxcox transformation
par(mfrow=c(1,1))
boxcox(latfit, lambda = seq(0, 4, 1/10))

#new fit for the latitude regression
boxcoxlatfit <- lm(latitude^3.43 ~ as.matrix(x))
summary(boxcoxlatfit)
boxcoxlatfitmse <- sum(boxcoxlatfit$residuals^2)/num_examples
boxcoxlatfitmse
par(mfrow=c(2,2))
plot(boxcoxlatfit)

#boxcox for longitude
par(mfrow=c(1,1))
boxcox(longfit)

##Ridge Regression
ridgelatitude <- cv.glmnet(as.matrix(x),latitude, alpha=0) 
plot(ridgelatitude)
min(ridgelatitude$cvm) #report minimum mse
ridgelatitude$lambda.min #report minimum lambda

ridgelongitude <- cv.glmnet(as.matrix(x),longitude, alpha=0) 
plot(ridgelongitude)
min(ridgelongitude$cvm) #report minimum mse
ridgelongitude$lambda.min #report minimum lambda

##Lasso Regression
lassolatitude <- cv.glmnet(as.matrix(x),latitude, alpha=1) 
plot(lassolatitude)
min(lassolatitude$cvm) #report minimum mse
lassolatitude$lambda.min #report minimum lambda

lassolongitude <- cv.glmnet(as.matrix(x),longitude, alpha=1) 
plot(lassolongitude)
min(lassolongitude$cvm) #report minimum mse
lassolongitude$lambda.min #report minimum lambda


##Elastic Net Regression
alphas <- c(.25,.5,.75)
latlambdas <- c()
latmses <- c()
longlambdas <- c()
longmses <- c()
for(a in alphas){
  enlatitude <- cv.glmnet(as.matrix(x),latitude, alpha=a)
  enlongitude <- cv.glmnet(as.matrix(x),longitude, alpha=a)
  latmses <- c(latmses, min(enlatitude$cvm)) #report minimum mse
  latlambdas <- c(latlambdas, enlatitude$lambda.min)
  longmses <- c(longmses, min(enlongitude$cvm)) #report minimum mse
  longlambdas <- c(longlambdas, enlongitude$lambda.min) #report minimum lambda
}

###Problem 2 - logistic regression

#setup
setwd("~/Documents/UIUC/CS 498/CS498MachineLearning/HW6")

#read in csv file
raw_data <- read.csv("creditcard.csv", skip=1, header = TRUE)
x <- as.matrix(raw_data[,seq(2,24)])
y <- as.factor(raw_data[,25])

#

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