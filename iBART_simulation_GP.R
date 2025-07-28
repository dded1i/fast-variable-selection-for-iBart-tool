set.seed(123)
library(ggplot2)
library(glmnet)

# Uncommon the following 3 lines if you want to see how iBART runs
# options(java.parameters = "-Xmx10g")
# library(bartMachine)
# source("R/BART_iter.R")

# Load necessary R files
source("R/descriptorGenerator.R")
source("R/GP_iter.R")
source("R/iBART.R")
source("R/L_zero_regression.R")
source("R/LASSO.R")
source("R/operations.R")
source("R/RF_iter.R")
source("R/utilis.R")

TruePositive <- function(names) {
  TP <- 0
  if (any(names == "((exp(x.1)-exp(x.2)))^2") | any(names == "abs((exp(x.1)-exp(x.2)))^2")) {
    TP <- TP + 1
  }
  if (any(names == "sin(pi*(x.3*x.4))")) {
    TP <- TP + 1
  }
  return(TP)
}

FalsePositive <- function(names, TP) {
  FP <- length(names) - TP
  return(FP)
}

FalseNegative <- function(TP) {
  FN <- 2 - TP
  return(FN)
}

Precision <- function(TP, FP) {
  if ((TP + FP) == 0) {
    return(0)
  } else {
    return(TP / (TP + FP))
  }
}

Recall <- function(TP, FN) {
  if ((TP + FN) == 0) {
    return(0)
  } else {
    return(TP / (TP + FN))
  }
}

F1 <- function(precision, recall) {
  if ((precision + recall) == 0) {
    return(0)
  } else{
    return(2 * precision * recall / (precision + recall))
  }
}

s <- 100 # Number of replicates
n <- 250 # Change n to 100 here to reproduce result in Supplementary Materials A.2.3
p <- 10  # Number of primary features
seeds <- sample.int(1000, size = 50)
seeds <- c(seeds, sample.int(1000, size = 50))

# Store your method's result here
iNP_results_og <- data.frame(TP = rep(0, s),
                             FP = rep(0, s),
                             TN = rep(0, s),
                             FN = rep(0, s),
                             precision = rep(0, s),
                             recall = rep(0, s),
                             F1 = rep(0, s))

# Store your method + L0 result here
iNP_results_aic <- data.frame(TP = rep(0, s),
                              FP = rep(0, s),
                              TN = rep(0, s),
                              FN = rep(0, s),
                              precision = rep(0, s),
                              recall = rep(0, s),
                              F1 = rep(0, s))

for (j in 1:s) {
  set.seed(seeds[j])
  ################################ Generate data ################################
  X <- matrix(runif(n * p, min = -1, max = 1), nrow = n, ncol = p)
  colnames(X) <- paste("x.", seq(from = 1, to = p, by = 1), sep = "")
  y <- 15 * (exp(X[, 1]) - exp(X[, 2]))^2 + 20 * sin(pi * X[, 3] * X[, 4]) + rnorm(n, mean = 0, sd = 0.5)
  
  iNP_model <- iBART(X = X, y = y,
                     name = colnames(X),
                     method = "GP",  # Change "BART-G.SE" to "GP" or "RF" here 
                     opt = c("unary", "binary", "unary"), 
                     sin_cos = TRUE,
                     apply_pos_opt_on_neg_x = FALSE,
                     Lzero = TRUE,
                     K = 4,
                     aic = TRUE,
                     standardize = FALSE,
                     seed = seeds[j])
  
  # original model
  iNP_results_og$TP[j] <- TruePositive(iNP_model$descriptor_names)
  iNP_results_og$FP[j] <- FalsePositive(iNP_model$descriptor_names, iNP_results_og$TP[j])
  iNP_results_og$FN[j] <- FalseNegative(iNP_results_og$TP[j])
  iNP_results_og$precision[j] <- Precision(iNP_results_og$TP[j], iNP_results_og$FP[j])
  iNP_results_og$recall[j] <- Recall(iNP_results_og$TP[j], iNP_results_og$FN[j])
  iNP_results_og$F1[j] <- F1(iNP_results_og$precision[j], iNP_results_og$recall[j])
  
  # AIC model
  iNP_results_aic$TP[j] <- TruePositive(iNP_model$Lzero_AIC_names)
  iNP_results_aic$FP[j] <- FalsePositive(iNP_model$Lzero_AIC_names, iNP_results_aic$TP[j])
  iNP_results_aic$FN[j] <- FalseNegative(iNP_results_aic$TP[j])
  iNP_results_aic$precision[j] <- Precision(iNP_results_aic$TP[j], iNP_results_aic$FP[j])
  iNP_results_aic$recall[j] <- Recall(iNP_results_aic$TP[j], iNP_results_aic$FN[j])
  iNP_results_aic$F1[j] <- F1(iNP_results_aic$precision[j], iNP_results_aic$recall[j])
  
  cat("Iteration: ", j, "/100... \n", sep = "")
}


# Combine all results
iNP_results_full <- rbind(iNP_results_og, iNP_results_aic)
iNP_results_full$Methods <- factor(rep(1:2, each = s), labels = c("iBART", "iBART+L0"))

# F1 plot
F1_median <- aggregate(F1 ~ Methods, iNP_results_full, median)
F1_median$F1 <- round(F1_median$F1, digits = 2)

F1_plot <- ggplot(iNP_results_full, aes(x = Methods, y = F1)) +
  geom_boxplot() +
  stat_summary(fun = median, colour = "darkred", geom = "point",
               shape = 18, size = 3, show.legend = FALSE) +
  geom_text(data = F1_median, aes(label = F1, y = F1),
            vjust = -0.4,
            colour = "blue") +
  scale_x_discrete(name = "Method") +
  scale_y_continuous(name = "F1 Scores") +
  theme(legend.position = "none")
F1_plot