set.seed(123)
library(cli)

# Load necessary R files
source("R/descriptorGenerator.R")
source("R/GP_iter.R")
source("R/operations.R")
source("R/utilis.R")

# Load iteration 1 results
load("results/iGP_iter1_result_updated.RData")

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

perf_update <- function(results, pos_name, pos_idx, j,  p) {
  results$TP[j] <- 0 # In case the initial value isn't 0
  if (any(c("(exp(x.1)-exp(x.2))", "abs((exp(x.1)-exp(x.2)))") %in% pos_name)) {
    results$TP[j] <- results$TP[j] + 1
    results$var1[j] <- TRUE
  }
  if ("(x.3*x.4)" %in% pos_name) {
    results$TP[j] <- results$TP[j] + 1
    results$var2[j] <- TRUE
  }
  results$FP[j] <- length(pos_idx) - results$TP[j]
  results$FN[j] <- 2 - results$TP[j]
  results$TN[j] <- p - results$TP[j] - results$FP[j] - results$FN[j]
  results$precision[j] <- Precision(results$TP[j], results$FP[j])
  results$recall[j] <- Recall(results$TP[j], results$FN[j])
  results$F1[j] <- F1(results$precision[j], results$recall[j])
  results$pos_idx[[j]] <- pos_idx
  
  return(results)
}

s <- 100 # Number of replicates
n <- 250 # Change n to 100 here to reproduce result in Supplementary Materials A.2.3
p <- 10  # Number of primary features
seeds <- sample.int(1000, size = 50)
seeds <- c(seeds, sample.int(1000, size = 50))

# Store your method's result here
iGP_results_iter2 <-  
  data.frame(TP = rep(0, s),
             FP = rep(0, s),
             TN = rep(0, s),
             FN = rep(0, s),
             precision = rep(0, s),
             recall = rep(0, s),
             F1 = rep(0, s),
             pos_idx = I(vector("list", length = 100)),
             var1 = rep(FALSE, s),
             var2 = rep(FALSE, s))

# GP parameters
alpha <- 0.25
a_r <- 2
b_r <- 0.1
total_iters <- 1500
pre_adapt_iters <- 100


for (j in 1:s) {
  set.seed(seeds[j])
  ################################ Generate data ################################
  X <- matrix(runif(n * p, min = -1, max = 1), nrow = n, ncol = p)
  colnames(X) <- paste("x.", seq(from = 1, to = p, by = 1), sep = "")
  y <- 15 * (exp(X[, 1]) - exp(X[, 2]))^2 + 20 * sin(pi * X[, 3] * X[, 4]) + rnorm(n, mean = 0, sd = 0.5)
  
  # Simulate correct selection of x.1, x.2, x.3, x.4
  X0 <- X[, 1:4]
  set.seed(seeds[j])
  dat0 <- list(X = X0,
               unit = NULL,
               name = colnames(X0))
  
  # Apply Unary operations to X0
  dat1 <- descriptorGenerator(data = dat0, opt = "unary", sin_cos = TRUE,
                              apply_pos_opt_on_neg_x = FALSE, verbose = FALSE)
  
  # Apply iter 1 selection
  idx_iter1 <- sort(iGP_results$pos_idx[j][[1]], decreasing = FALSE)
  dat1$X <- dat1$X[, idx_iter1]
  dat1$name <- dat1$name[idx_iter1]
  
  # Combind X0 and X1
  dat1$X <- cbind(dat0$X, dat1$X)
  dat1$name <- c(dat0$name, dat1$name)
  
  # Remove duplicated columns
  dup_index <- duplicated(round(dat1$X, digits = 10), MARGIN = 2)
  dat1$X <- dat1$X[, !dup_index]
  dat1$name <- dat1$name[!dup_index]
  
  # Apply Binary operations to X1
  dat2 <- descriptorGenerator(data = dat1, opt = "binary", sin_cos = TRUE,
                              apply_pos_opt_on_neg_x = FALSE, verbose = FALSE)
  colnames(dat2$X) <- dat2$name
  
  ## GP
  p2 <- ncol(dat2$X)
  n2 <- nrow(dat2$X)
  
  # Initialization
  gp <- list(
    y_train = y,
    alpha = array(alpha, p2),
    gamma = runif(p2) < alpha,
    ro = array(1, p2),
    lambda_a = rgamma(1, shape = 1),
    lambda_z = rgamma(1, shape = 1),
    r = rgamma(1, shape = a_r, scale = b_r)
  )
  
  gp$ro[gp$gamma] <- runif(sum(gp$gamma))
  
  x_expanded <- aperm(array(dat2$X, c(n2, p2, n2)), c(2, 1, 3))
  dist_sq <- (x_expanded - aperm(x_expanded, c(1, 3, 2)))^2
  gp$dist_squares <- dist_sq
  gp$G <- apply(dist_sq * array(log(gp$ro), c(p2, n2, n2)), c(2, 3), sum)
  gp <- GP_calc_log_likelihood(gp)
  
  # Training
  cli_progress_bar("GP training", total = total_iters)
  for (i in 1:total_iters) {
    gp <- GP_train_iter(gp, i, pre_adapt_iters, p2)
    cli_progress_update()
  }
  cli_progress_done()
  
  # Training results
  pos_idx <- which(gp$gamma)
  pos_name <- dat2$name[pos_idx]
  
  # Store result
  iGP_results_iter2 <- perf_update(iGP_results_iter2, pos_name, pos_idx, j, ncol(dat2$X))
  
  cat("Iteration: ", j, "/100... \n", sep = "")
  save(iGP_results_iter2, file = "iGP_iter2_result.RData")
}

