##Neeraj Asthana (nasthan2)
##CS 498 HW4

##Environment Setup
library(lattice)
setwd('/home/neeraj/Documents/UIUC/CS 498/CS498MachineLearning/HW4')

##Problem 3.4
#Read data
iris_data <- read.csv("iris.data", header=FALSE)
names <- c("sep_len", "sep_width", "petal_len", "petal_width", "class")
colnames(iris_data) <- names

##Problem 3.4a
#Scatterplot matrix
pchr <- c(1,2,3)
colors <- c("red", "green", "blue")
species_names <- c("Setosa","Versicolor","Virginica")
feature_names <- c("Sepal Length", "Sepal Width", "Petal Length", "Petal Width")

ss <- expand.grid(species=1:3)
parset <- with( ss, simpleTheme(pch=pchr[species],col=colors[species]))

splom(iris_data[,c(1:4)], groups=iris_data$class,
      varnames = feature_names,
      par.settings=parset,
      #panel=panel.superpose, 
      key=list(title="Species of Iris",
               columns=3,
               points=list(pch=pchr),
               text=list(species_names)))

##Problem 3.4b - Now obtain the first two principal components of the data. Plot the data on those two principal components alone, again showing each species with a different marker. Has this plot introduced significant distortions? Explain
iris_features <- iris_data[,c(1,2,3,4)]
pca <- prcomp(iris_features, center = TRUE, scale. = TRUE)
secondpca <- princomp(iris_featues, cor = TRUE, scores= TRUE)
plot(secondpca$scores[,c(1,2)], col=iris_data[,5])

##Problem 3.4c - Now use PLS1 to obtain two discriminative directions, and project thedata on to those directions. Does the plot look better? Explain Keep inmind that the most common error here is to forget that the X and the Yin PLS1 are centered - i.e. we subtract the mean.


##Problem 3.5

##Problem 3.7