

##Problem 1

#Constants
num_documents <- 1500 
num_vocab_words <- 12419 
num_words <- 746316
topics <- 30

#Setup
#read in files
setwd("/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW5")
vocab <- read.csv("vocab.nips.txt")
documents <- read.csv("docword.nips.txt", sep = " ", skip = 3, header = FALSE)
colnames(documents) <- c("document", "word_id", "count")

#create documents
vecs <- matrix(0,num_documents,num_vocab_words)
for(i in 1:num_words){
  vecs[documents[i,1], documents[i,2]] = documents[i,3]
}

#set initial values for pis and p's (probs in this code)
pis <- matrix(1/topics,1,topics)
probs <- matrix(0,topics,num_vocab_words)
#set probs to be random values that sum to 1 for each topic 
for(i in 1:topics){
  x <- runif(num_vocab_words)
  rowvals <- x / sum(x)
  probs[i,] <- rowvals
}


#E Step
#function to calculate the expected value of log liklihood:
logliklihood <- function(){
  inner <- vecs %*% t(log(probs))
  woweights <- matrix(0,num_documents, topics)
  #add logs of the pis
  for(i in seq(topics)){
    woweights[,i] <- inner[,i] + log(pis[i])
  }
  #calculate w_ij
  
}

#M Step

#EM Combined


#Problem 2