##Neeraj Asthana (nasthan2)
##CS 498 HW3

#environment
options(warn=-1)
library(caret)

#load data
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW3')
raw_train_data<-read.csv('pubfig_train_50000_pairs.txt', sep = "\t")
