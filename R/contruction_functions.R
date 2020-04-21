#' A reparameterization in order to sample from the learned standard normal distribution of the VAE
#'
#' @param arg a layer of tensors representing the mean and variance
sampling_standard_normal <- function(arg){
  num_skills <- keras::k_int_shape(arg)[[2]] / 2
  z_mean <- arg[, 1:num_skills]
  z_log_var <- arg[, (num_skills + 1):(2 * num_skills)]
  b_size <- keras::k_int_shape(z_mean)[[1]]
  eps <- keras::k_random_normal(
    shape = c(b_size, keras::k_cast(num_skills, dtype = 'int32')),
    mean = 0, stddev = 1
  )
  z_mean + tensorflow::tf$multiply(keras::k_exp(z_log_var / 2), eps)
}

#' A reparameterization in order to sample from the learned multivariate normal distribution of the VAE
#'
#' @param arg a layer of tensors representing the mean and log cholesky transform of the covariance matrix
sampling_normal_full_covariance <- function(arg){ #TODO: This is doing something wrong - i think it is fixed
  num_skills <- as.integer(-1.5 + sqrt(2 * keras::k_int_shape(arg)[[2]] + 9/4))
  z_mean <- arg[, 1:(num_skills)]
  b_size <- keras::k_int_shape(z_mean)[[1]]
  if (is.null(b_size)){ #fix for batch size and matmul
    b_size <- 1
  }
  #TODO:remove this comment block when I'm sure everything is correct
  # this was when using tf$contrib (TF version <= 1.9)
  # z_log_cholesky <- tensorflow::tf$contrib$distributions$fill_triangular(
    # arg[1:b_size, (num_skills + 1):(keras::k_int_shape(arg)[[2]])])
  b <- tfprobability::tfb_fill_triangular(upper=FALSE) # this works for TF version >=2.1
  z_log_cholesky <- b$forward(
    arg[1:b_size, (num_skills + 1):(keras::k_int_shape(arg)[[2]])])
  z_cholesky <- tensorflow::tf$linalg$expm(z_log_cholesky)
  eps <- keras::k_random_normal(
    shape = c(b_size, num_skills, 1),
    mean = 0, stddev = 1
  )
  z_mean + keras::k_reshape(tensorflow::tf$matmul(z_cholesky, eps), shape = c(-1, num_skills))
}

#' A custom kernel constraint function that restricts weights between the learned distribution and output. Nonzero weights are determined by the Q matrix
#'
#' @param Q a binary matrixof size \code{num_skills} by \code{num_items}
q_constraint <- function(Q){ #note - might be able to use custom layer class to implement this instead
  constraint <- function(w){
    target <- w * Q
    diff = w - target
    w <- w * keras::k_cast(keras::k_equal(diff, 0), keras::k_floatx()) # enforce Q-matrix connections
    w * keras::k_cast(keras::k_greater_equal(w, 0), keras::k_floatx()) # require non-negative weights
  }
  constraint
}

q_1pl_constraint <- function(Q){
  constraint <- function(w){
    Q # require all weights = 1 according to Q matrix so VAE will esimate 1-parameter logistic model
  }
}
