##############################
## GP_iter helper functions ##
##############################

library(msos)  # for the logdet() function

GP_calc_log_likelihood <- function(gp) {
  n <- length(gp$y_train)
  cov_matrix <- diag(1 / gp$r, n) + 1 / gp$lambda_a + exp(gp$G) / gp$lambda_z
  if (rcond(cov_matrix) <= 1e-10) {
    gp$log_likelihood <- NA
    return(gp)
  }
  const_term <- n * log(2 * pi)
  det_term <- logdet(cov_matrix)
  inv_term <- (gp$y_train %*% solve(cov_matrix, gp$y_train))[1]
  gp$log_likelihood <- -(const_term + det_term + inv_term) / 2
  return(gp)
}

GP_update_ro <- function(gp, pos, new_value) {
  old_value <- gp$ro[pos]
  gp$G <- gp$G + gp$dist_squares[pos,,] * log(new_value / old_value)
  gp$ro[pos] <- new_value
  return(gp)
}

GP_train_iter <- function(gp, iter, pre_adapt_iters) {
  # Update gamma and ro
  for (pos in 1:p) {
    gp_next <- gp

    # Between models
    if (!gp_next$gamma[pos] && (
      iter <= pre_adapt_iters || runif(1) < gp$alpha[pos]
    )) {
      gp_next$gamma[pos] <- TRUE
      gp_next <- GP_calc_log_likelihood(GP_update_ro(gp_next, pos, runif(1)))
      if (is.na(gp_next$log_likelihood)) next
      accept_prob <- exp(3 * (gp_next$log_likelihood - gp$log_likelihood))
      if (!is.nan(accept_prob) && runif(1) < accept_prob) {
        gp <- gp_next
      }

    } else if (gp_next$gamma[pos] && (
      iter <= pre_adapt_iters || runif(1) >= gp$alpha[pos]
    )) {
      gp_next$gamma[pos] <- FALSE
      gp_next <- GP_calc_log_likelihood(GP_update_ro(gp_next, pos, 1))
      if (is.na(gp_next$log_likelihood)) next
      accept_prob <- exp(3 * (gp_next$log_likelihood - gp$log_likelihood))
      if (!is.nan(accept_prob) && runif(1) < accept_prob) {
        gp <- gp_next
      }
    }

    # Within the model
    if (gp_next$gamma[pos]) {
      gp_next <- gp
      gp_next <- GP_calc_log_likelihood(GP_update_ro(gp_next, pos, runif(1)))
      if (is.na(gp_next$log_likelihood)) next
      accept_prob <- exp(3 * (gp_next$log_likelihood - gp$log_likelihood))
      if (!is.nan(accept_prob) && runif(1) < accept_prob) {
        gp <- gp_next
      }
    }
  }

  # Update lambda_a
  gp_next <- gp
  gp_next$lambda_a <- rgamma(1, shape = 1, scale = gp_next$lambda_a)
  gp_next <- GP_calc_log_likelihood(gp_next)
  q_div_q1 <- dgamma(gp$lambda_a, gp_next$lambda_a) /
              dgamma(gp_next$lambda_a, gp$lambda_a)
  if (is.na(gp_next$log_likelihood)) return(gp)
  accept_prob <- exp(3 * (gp_next$log_likelihood - gp$log_likelihood)) * q_div_q1
  if (!is.nan(accept_prob) && runif(1) < accept_prob) {
    gp <- gp_next
  }

  # Update lambda_z
  gp_next <- gp
  gp_next$lambda_z <- rgamma(1, shape = 1, scale = gp_next$lambda_z)
  gp_next <- GP_calc_log_likelihood(gp_next)
  q_div_q1 <- dgamma(gp$lambda_z, gp_next$lambda_z) /
              dgamma(gp_next$lambda_z, gp$lambda_z)
  if (is.na(gp_next$log_likelihood)) return(gp)
  accept_prob <- exp(3 * (gp_next$log_likelihood - gp$log_likelihood)) * q_div_q1
  if (!is.nan(accept_prob) && runif(1) < accept_prob) {
    gp <- gp_next
  }

  # Update r
  gp_next <- gp
  gp_next$r <- rgamma(1, shape = 1, scale = gp_next$r)
  gp_next <- GP_calc_log_likelihood(gp_next)
  q_div_q1 <- dgamma(gp$r, gp_next$r) /
              dgamma(gp_next$r, gp$r)
  if (is.na(gp_next$log_likelihood)) return(gp)
  accept_prob <- exp(3 * (gp_next$log_likelihood - gp$log_likelihood)) * q_div_q1
  if (!is.nan(accept_prob) && runif(1) < accept_prob) {
    gp <- gp_next
  }

  # Finishing
  if (iter > pre_adapt_iters) {
    gp$alpha <- (gp$alpha * (iter - 1) + gp$gamma) / iter
  }
  return(gp)
}

###############################

GP_iter <- function(data = NULL,
                    ########################
                    alpha = NULL,
                    a_r = NULL,
                    b_r = NULL,
                    total_iters = NULL,
                    pre_adapt_iters = NULL,                    
                    ########################
                    standardize = TRUE,
                    train_idx = NULL,
                    seed = NULL,
                    iter = 1) {
  data$X <- as.matrix(data$X) # Make sure X is a matrix, not a vector
  if (!is.null(data$X_selected)) data$X_selected <- as.matrix(data$X_selected)
  dat <- trainingSplit(X = data$X, y = data$y, train_idx = train_idx)
  
  
  ##########################################
  #### Your implementation starts here #####
  # 1. Training data are dat$X_train (matrix object) and dat$y_train (numerical vector)
  # 2. Train GP on dat$X_train and dat$y_train
  # 3. Record pos_idx <- selected column indices of dat$X_train
  
  X <- dat$X_train
  p <- ncol(X)
  n <- nrow(X)

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

  x_expanded <- aperm(array(X, c(n, p, n)), c(2, 1, 3))
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
  
  ##### Your implementation ends here ######
  ##########################################
  
  # length(pos_idx) == 0 means no feature is selected
  if (length(pos_idx) == 0) {
    cat(paste0("BART didn't select anything in Iteration ", iter, "..."))
    cat("Proceeding to the next iteration...")
    data$no_sel_count <- data$no_sel_count + 1
    data$iBART_sel_size <- c(data$iBART_sel_size, NA)
    return(data) # early stop
  }
  
  # If BART selected some variables...
  # Union new selections with previous selections
  data$X_selected <- cbind(data$X_selected, data$X[, pos_idx])
  data$name_selected <- c(data$name_selected, data$name[pos_idx])
  if (!is.null(data$unit)) data$unit_selected <- cbind(data$unit_selected, data$unit[, pos_idx])
  
  # Remove duplicated data
  dup_index <- duplicated(data$X_selected, MARGIN = 2)
  data$X <- data$X_selected <- as.matrix(data$X_selected[, !dup_index])
  data$name <- data$name_selected <- data$name_selected[!dup_index]
  if (!is.null(data$unit)) data$unit <- data$unit_selected <- as.matrix(data$unit_selected[, !dup_index])
  
  # Attach colnames in case ncol(data$X_selected) == 1
  colnames(data$X_selected) <- colnames(data$X) <- data$name
  data$iBART_sel_size <- c(data$iBART_sel_size, length(pos_idx))
  return(data)
}

