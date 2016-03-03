##Neeraj Asthana (nasthan2)
##CS 498 HW4

##Environment Setup
library(lattice)
library(plsdepot)
library(ggplot2)
library(scatterplot3d)
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
      key=list(title="Species of Iris",
               columns=3,
               points=list(pch=pchr),
               text=list(species_names)))

##Problem 3.4b
title34b <- "Problem 3.4b - Iris Data Projected onto First 2 Principal Components"
iris_features <- iris_data[,c(1,2,3,4)]
pca <- prcomp(iris_features, center = TRUE, scale. = TRUE)
plot(pca$x[,1], pca$x[,2], col=iris_data[,5], xlab = "Principal Component 1", ylab = "Principal Component 2", main=title34b)

##Problem 3.4c
title34c <- "Problem 3.4b - Iris Data Projected onto First 2 Discriminative Directions"
hot_vectors <- matrix(0, 150, 3)
hot_vectors[1:50,1] <- 1
hot_vectors[51:100,2] <- 1
hot_vectors[101:150,3] <- 1
pls1 <- plsreg2(iris_features, hot_vectors)
plot(pls1$x.scores[,1], pls1$x.scores[,2], col=iris_data[,5], main = title34c)






##Problem 3.5
wine_data <- read.csv("wine.data", header=FALSE)
wine_features <- wine_data[,-1]
wine_labels <- wine_data[,1]

##Problem 3.5a
title35a <- "Problem 3.5a - Wine Data Eigenvalues"
covmat <- cov(wine_features)
eig <- eigen(covmat, symmetric=FALSE)
print(eig$values)
plot(eig$values, type="b", ylab="Eigenvalue", xlab = "Eigenvalue Index", main=title35a)

##Problem 3.5b
title35b1 <- "Problem 3.5b - Wine Data EigenVector Stem Plot"
counts <- t(eig$vectors[,1:3])
barplot(counts, width=.1, space=1, col=colors, legend = c("Component 1", "Component 2", "Component 3"), main=title35b1)
title35b2 <- "Problem 3.5b - Wine Data Stem Plot"
princip2 <- prcomp(wine_features, center = TRUE, scale. = TRUE, xlab="Index", main=title35b2)
barplot(t(princip2$x[,1:3]), width=.1, space=1, col=colors, legend = c("Component 1", "Component 2", "Component 3"), args.legend = list(x="bottomright") )

##Problem 3.5c
title35c <- "Problem 3.5c - Wine Data Projected onto the first 2 Principal Components"
plot(princip2$x[,1:2],col="white", pch=3, xlab = "Principal Component 1", ylab = "Principal Component 2", main=title35c)
text(princip2$x[,1:2], col=wine_labels, labels = wine_labels)






##Problem 3.7
cancer_data <- read.csv("wdbc.data", header = FALSE)
cancer_id <- cancer_data[,1]
cancer_labels <- cancer_data[,2]
cancer_features <- cancer_data[,-c(1,2)]

##Problem 3.7a
princip3 <- prcomp(cancer_features, center=TRUE, scale. = TRUE)
title37a <- "Problem 3.7a - 3d plot of first 3 Principal Components"
plot3d <- scatterplot3d(princip3$x[cancer_labels == "M",1:3], color = "red", angle=110, main = title37a)
plot3d$points3d(princip3$x[cancer_labels != "M",1:3], col = "blue")

##Problem 3.7b
cancer_hot_vectors <- matrix(0, 569, 2)
cancer_hot_vectors[cancer_labels == "B",1] <- 1
cancer_hot_vectors[cancer_labels == "M",2] <- 1
cancer_pls1 <- plsreg2(cancer_features, cancer_hot_vectors, comps=3)
title37b <- "Problem 3.7a - 3d plot of breast cancer data projected onto the first 3 Discriminative Directions"
plot3d2 <- scatterplot3d(cancer_pls1$x.scores[cancer_labels == "M",1:3], color = "red", angle=40, main = title37b)
plot3d$points3d(cancer_pls1$x.scores[cancer_labels != "M",1:3], col = "blue")