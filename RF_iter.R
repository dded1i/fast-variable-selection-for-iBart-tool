RF_iter <- function(data = NULL,
                    ########################
                    ## RF parameters here ##
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
  # 2. Train RF on dat$X_train and dat$y_train
  # 3. Record pos_idx <- selected column indices of dat$X_train
  
  
  
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
