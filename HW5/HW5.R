

library(matrixStats)
library(jpeg)

##Problem 1

#Constants
num_documents <- 1500 
num_vocab_words <- 12419 
num_words <- 746316
topics <- 30
smoothing_constant <- .00025
stop_criteria <- .001

#Setup
#read in files
setwd("/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW5")
vocab <- unlist(read.csv("vocab.nips.txt",stringsAsFactors = FALSE))
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

Qs <- c()
while(TRUE){
  #E Step - calculate the expected value of log liklihood:
  inner <- vecs %*% t(log(probs)) #[1500*30] sums of features multiplied by probs for each doc and cluster
  woweights <- matrix(0,num_documents, topics) #inner + probablity of each cluster 
  #add logs of the pis
  for(i in seq(topics)){
    woweights[,i] <- inner[,i] + log(pis[i])
  }
  #calculate w_ij s
  wijs <- matrix(0,num_documents, topics)
  Ajs <- woweights
  rowmax <- apply(woweights, 1, max)
  thirdterms <- matrix(0,num_documents)
  for(i in seq(num_documents)){
    thirdterms[i] <- logSumExp(Ajs[i,] - rowmax[i])
  }
  w <- Ajs - unlist(as.list(rowmax - thirdterms))
  unnormalwijs <- exp(w) #must normalize these weights to sum to 1
  for(i in seq(num_documents)){
    wijs[i,] = unnormalwijs[i,] / sum(unnormalwijs[i,])
  }
  #final multiplication and sum
  finalvals <- woweights*wijs
  Q <- sum(finalvals)
  print(Q)
  Qs <- c(Qs, Q)
  
  #M Step - update pis and probs
  for(j in seq(topics)){
    #Update p's with additive smoothing
    top <- colSums(vecs * wijs[,j]) + smoothing_constant
    bottom <- sum(rowSums(vecs) * wijs[,j]) + (smoothing_constant * num_vocab_words)
    probs[j,] <-top/bottom
    
    #update pis
    pis[j] <- sum(wijs[,j]) / num_documents
  }
  
  #stopping rule
  if(length(Qs) > 1){
    if(Q - Qs[length(Qs)-1] < stop_criteria){
      break
    }
  }
}

#pis plot
plot(unlist(as.list(pis)), type='l', ylab = "probability", xlab="topic", main = "Probability a Topic is Selected")

tab <- c()
#10 highest occuring words per topic
for(i in seq(topics)){
  max10 <- sort(probs[i,], decreasing = TRUE)[10]
  tab <- c(tab, vocab[which(probs[i,] >= max10)][1:10])
  print(paste("Topic", i , ":"))
}
tab <- matrix(tab, nrow = 30)
rownames(tab) <- paste("Topic", seq(topics))






#Problem 2
#read in images
balloons <- readJPEG("balloons.jpg")
mountains <- readJPEG("mountains.jpg")
nature <- readJPEG("nature.jpg")
ocean <- readJPEG("ocean.jpg")
polarlights <- readJPEG("polarlights.jpg")

empic <- function(pic, clusters){
  stop_criteria <- .00001
  num_pixels <- prod(dim(pic))/3
  height <- dim(pic)[1]
  width <- dim(pic)[2]
  
  #get a set of all of the pixels together
  pixels <- matrix(0,num_pixels,3)
  for(i in seq(height)){
    for(j in seq(width)){
      pixels[((i-1)*width)+j,] <- pic[i,j,]
    }
  }
  
  tesing <- array(0,c(height, width,3))
  for(i in seq(height)){
    for(j in seq(width)){
      tesing[i,j,] <- pixels[(i-1)*width+j,]
    }
  }
  
  pixels<-scale(pixels)
  
  pis <- matrix(1/clusters,1,clusters)
  ranvals <- runif(3*clusters)
  #set means to be random values that sum to 1 for each cluster mean
  means <- matrix(ranvals, nrow=clusters)
  
  #EM steps
  Qs <- c()
  while(TRUE){
    #E Step - calculate the expected value of log liklihood:
    inner <- matrix(0,num_pixels, clusters)
    for(i in seq(clusters)){
      dist <- t(t(pixels)-means[i,])
      inner[,i] <- (-.5) * rowSums(dist^2)
    }
    #calculate wijs
    top <- exp(inner) %*% diag(pis[1:clusters])
    bottom <- rowSums(top)
    wijs <- top/bottom
    #calculate Q
    Q <- sum(inner*wijs)
    print(Q)
    Qs <- c(Qs, Q)
    
    #M step
    for(j in seq(clusters)){
      #Update p's with additive smoothing
      top <- colSums(pixels * wijs[,j]) #+ smoothing_constant
      bottom <- sum(wijs[,j]) #+ (smoothing_constant * num_pixels)
      means[j,] <-top/bottom
      
      #update pis
      pis[j] <- sum(wijs[,j]) / num_pixels
    }
    
    #stopping rule
    if(length(Qs) > 1){
      if(Q - Qs[length(Qs)-1] < stop_criteria){
        break
      }
    }
  }
  
  #Construct final image
  final_img <- array(0,c(height, width,3))
  for(i in seq(height)){
    for(j in seq(width)){
      index <- (i-1)*width + j
      point <- pixels[index,]
      meanseg <- which(wijs[index,] == max(wijs[index,]))
       final_img[i,j,] <- means[meanseg,]*attr(pixels, 'scaled:scale') + attr(pixels, 'scaled:center')
    }
  }
  return(final_img)
}

writeJPEG(empic(balloons,10), "balloons_segmented10.jpg",quality = 1)
writeJPEG(empic(balloons,20), "balloons_segmented20.jpg",quality = 1)
writeJPEG(empic(balloons,50), "balloons_segmented50.jpg",quality = 1)

writeJPEG(empic(mountains,10), "mountains_segmented10.jpg",quality = 1)
writeJPEG(empic(mountains,20), "mountains_segmented20.jpg",quality = 1)
writeJPEG(empic(mountains,50), "mountains_segmented50.jpg",quality = 1)

writeJPEG(empic(nature,10), "nature_segmented10.jpg",quality = 1)
writeJPEG(empic(nature,20), "nature_segmented20.jpg",quality = 1)
writeJPEG(empic(nature,50), "nature_segmented50.jpg",quality = 1)

writeJPEG(empic(ocean,10), "ocean_segmented10.jpg",quality = 1)
writeJPEG(empic(ocean,20), "ocean_segmented20.jpg",quality = 1)
writeJPEG(empic(ocean,50), "ocean_segmented50.jpg",quality = 1)

writeJPEG(empic(polarlights,20), "polarlights_segmented201.jpg",quality = 1)
writeJPEG(empic(polarlights,20), "polarlights_segmented202.jpg",quality = 1)
writeJPEG(empic(polarlights,20), "polarlights_segmented203.jpg",quality = 1)
writeJPEG(empic(polarlights,20), "polarlights_segmented204.jpg",quality = 1)
writeJPEG(empic(polarlights,20), "polarlights_segmented205.jpg",quality = 1)