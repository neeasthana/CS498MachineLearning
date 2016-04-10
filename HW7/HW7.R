##Neeraj Asthana (nasthan2)
##CS 498 HW7

library(glmnet)

#setup
setwd("/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW7")

#read in data
locations <- read.csv("Locations.txt", header = TRUE, sep=" ")
temps <- read.csv("Oregon_Met_Data.txt", header = TRUE, sep = " ")

srange <- c(10000 , 150000 , 200000 , 250000 , 300000 , 350000)

##Problem 1
xmat = locations[,c(7,8)]
spaces<- dist(xmat , method = "euclidean",diag= FALSE,upper= FALSE)
msp <- as.matrix(spaces)


wmat <- exp(-msp/(2*srange[1]^2))


for( i in 2:6){
  grammmat < − exp(−msp/ ( 2 ∗ s r a n g e [ i ] ˆ 2 ) )wmat < −cbind ( wmat , grammmat )
}
##Problem 2

##Problem 3

##Extra Credit