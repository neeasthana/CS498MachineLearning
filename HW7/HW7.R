##Neeraj Asthana (nasthan2)
##CS 498 HW7

library(glmnet)

#setup
setwd("/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW7")

#read in data and grab relevant columns
raw_locations <- read.csv("Locations.txt", header = TRUE, sep=" ")
raw_temps <- read.csv("Oregon_Met_Data.txt", header = TRUE, sep = " ")
locations <- raw_locations[,c(1,7,8)]
temps <- raw_temps[,c(1,4,5,6)]

#remove all invalid temperatures
temps <- temps[temps$Tmin_deg_C != 9999,]

#produce constants for easy reference
n <- dim(locations)[1]
m <- dim(locations)[2]+1
gridsize <- 100

#create matrix to store all training values
x <- matrix(0,n,m)
colnames(x) <- c(colnames(locations), colnames(temps)[4])
x[,1] <- locations[,1]
x[,2] <- locations[,2]
x[,3] <- locations[,3]

#calculate the minimum temperature averages for each station which 
meanTemps <- tapply(temps$Tmin_deg_C, temps$SID, mean)
x[,4] <- meanTemps


##Problem 1 - simple nonparametric regression
#setup response grid
xmin <- min(xmat[,1])
xmax <- max(xmat[,1])
ymin <- min(xmat[,2])
ymax <- max(xmat[,2])
xvec <- seq(xmin,xmax,length=gridsize)
yvec <- seq(ymin,ymax,length=gridsize)


#setup predictors and response
xmat <- x[,2:3]
y <- x[,4]

#creates 6 different scale values based distances in msp
srange <- seq(65000,250000, 35000)


for(s in srange){

  #Create a matrix with all of the nonparameterized weights
  #All rows now sum to 1 after the kernel function is applied
  spaces<- dist(xmat , method = "euclidean",diag= FALSE,upper= FALSE)
  msp <- as.matrix(spaces)
  wmat <- exp(-msp^2/(2*s^2))
  
  #nonparam <- wmat/rowSums(wmat)
  model <- cv.glmnet(wmat, as.vector(y), alpha = 1, lambda = c(0,1))
  modelmse <- model$cvm[2]
  
  #create smoothing points to then be able to create plot
  predictionMat <- matrix(NA, gridsize^2, 2)
  for(i in 1:gridsize)
    for(j in 1:gridsize)
      predictionMat[(i-1)*gridsize + j, ] <- c(xvec[i], yvec[j])
  
  #create matrix of points that the kernel function has already evaluated
  diff_ij <- function(i,j){sqrt(rowSums((predictionMat[i,]-xmat[j,]) ^2 )) }
  sampledists <- outer(1:gridsize^2, 1:n, diff_ij)
  samplewmat <- exp(-sampledists^2/(2*s^2))
  
  #create predictions at those points
  predictions <- predict.cv.glmnet(model, samplewmat, s = 0)
  
  #generate the final grid for this model with specified scaling parameter
  finalgrid <- matrix(0,gridsize, gridsize)
  for(i in 1:gridsize)
    for(j in 1:gridsize)
      finalgrid[i,j] <- predictions[(i-1)*gridsize + j]
}


##Problem 2

##Problem 3

##Extra Credit