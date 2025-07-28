set.seed(123)

# Load necessary R files
source("R/descriptorGenerator.R")
source("R/GP_iter.R")
source("R/operations.R")
source("R/utilis.R")

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

perf_update <- function(results, j, idx0, pos_idx) {
  results$TP[j] <- length(intersect(idx0, pos_idx))
  results$FP[j] <- length(setdiff(pos_idx, idx0))
  results$FN[j] <- length(setdiff(idx0, pos_idx))
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
iGP_results <- data.frame(TP = rep(0, s),
                          FP = rep(0, s),
                          TN = rep(0, s),
                          FN = rep(0, s),
                          precision = rep(0, s),
                          recall = rep(0, s),
                          F1 = rep(0, s),
                          pos_idx = I(vector("list", length = 100)))


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
                              apply_pos_opt_on_neg_x = FALSE)
  X1 <- dat1$X
  colnames(X1) <- dat1$name
  idx0 <- which(dat1$name %in% c("x.3", "x.4", "exp(x.1)", "exp(x.2)"))
  
  ## GP
  p <- ncol(X1)
  n <- nrow(X1)
  
  # Initialization
  gp <- list(
    y_train = y,
    alpha = array(alpha, p),
    gamma = runif(p) < alpha,
    ro = array(1, p),
    lambda_a = rgamma(1, shape = 1),
    lambda_z = rgamma(1, shape = 1),
    r = rgamma(1, shape = a_r, scale = b_r)
  )
  
  gp$ro[gp$gamma] <- runif(sum(gp$gamma))
  
  x_expanded <- aperm(array(X1, c(n, p, n)), c(2, 1, 3))
  dist_sq <- (x_expanded - aperm(x_expanded, c(1, 3, 2)))^2
  gp$dist_squares <- dist_sq
  gp$G <- apply(dist_sq * array(log(gp$ro), c(p, n, n)), c(2, 3), sum)
  gp <- GP_calc_log_likelihood(gp)
  
  # Training
  for (i in 1:total_iters) {
    gp <- GP_train_iter(gp, i, pre_adapt_iters)
  }
  
  # Training results
  pos_idx <- (1:p)[gp$gamma]
  
  # Store results
  iGP_results <- perf_update(iGP_results, j, idx0, pos_idx)
  
  cat("Iteration: ", j, "/100... \n", sep = "")
  save(iGP_results, file = "iGP_iter1_result.RData")
}

