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

#preform logistic regression using cross 