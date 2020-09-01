#' A custom loss function for a VAE learning a standard normal distribution
#'
#' @param encoder the encoder model of the VAE, used to obtain z_mean and z_log_var from inputs
#' @param kl_weight weight for the KL divergence term
#' @param rec_dim the number of nodes in the input/output of the VAE
vae_loss_standard_normal <- function(encoder, kl_weight, rec_dim){
  loss <- function(input, output){
    vals <- encoder(input)
    z_mean_val <- vals[[1]]
    z_log_var_val <- vals[[2]]
    kl_loss <- 0.5 * keras::k_sum(keras::k_square(z_mean_val) +
                                    keras::k_exp(z_log_var_val) -
                                    1 -
                                    z_log_var_val,
                                    axis = -1L)
    rec_loss <- rec_dim * keras::loss_binary_crossentropy(input, output)
    # rec_loss + kl_weight * kl_loss
    keras::k_mean(kl_weight * kl_loss + rec_loss)
  }
  loss
}

#' A custom loss function for a VAE learning a multivariate normal distribution with a full covariance matrix
#'
#' @param encoder the encoder model of the VAE, used to obtain z_mean and z_log_cholesky from inputs
#' @param inv_skill_cov a constant tensor matrix of the inverse of the covariance matrix being learned
#' @param det_skill_cov a constant tensor scalar representing the determinant of the covariance matrix being learned
#' @param skill_mean a constant tensor vector representing the means of the latent skills being learned
#' @param kl_weight weight for the KL divergence term
#' @param rec_dim the number of nodes in the input/output of the VAE
vae_loss_normal_full_covariance <- function(encoder,
                                            inv_skill_cov,
                                            det_skill_cov,
                                            skill_mean,
                                            kl_weight,
                                            rec_dim){
  loss <- function(y_true, y_pred){
    vals <- encoder(y_true)
    z_mean <- vals[[1]]
    z_log_chol_vals <- vals[[2]]
    b <- tfprobability::tfb_fill_triangular(upper=FALSE)
    z_log_cholesky <- b$forward(z_log_chol_vals)
    z_cholesky <- tensorflow::tf$linalg$expm(z_log_cholesky, name='expm_loss') #maybe issue here
    z_cov_matrix <- tensorflow::tf$matmul(z_cholesky, tensorflow::tf$transpose(z_cholesky, c(0L, 2L, 1L)))
    num_skills <- keras::k_int_shape(skill_mean)[[2]]
    diff <- tensorflow::tf$reshape(z_mean - skill_mean, c(-1L, num_skills, 1L))
    temp <- keras::k_dot(tensorflow::tf$transpose(diff, c(0L, 2L, 1L)), inv_skill_cov)
    kl_loss <- 0.5 * (tensorflow::tf$linalg$trace(tensorflow::tf$transpose(
                          keras::k_dot(inv_skill_cov, z_cov_matrix), c(1L,0L,2L))) +
                        keras::k_reshape(tensorflow::tf$matmul(temp, diff), c(-1)) -
                        tensorflow::tf$constant(num_skills, dtype = 'float32') +
                        log(det_skill_cov / tensorflow::tf$linalg$det(z_cov_matrix)))
    rec_loss <- rec_dim * keras::loss_binary_crossentropy(y_true, y_pred)
    keras::k_mean(kl_weight * kl_loss + rec_loss)
  }
  loss
}

#' This function is deprecated and will be removed in future versions. A loss function for a VAE learning a standard normal distribution for use with custom training
#'
#' @param z_mean a tensor (layer) representing the mean in the VAE
#' @param z_log_var a tensor (layer) representing the log variance in the VAE
#' @param kl_weight weight for the KL divergence term
#' @param rec_dim the number of nodes in the input/output of the VAE
experimental_standard_loss <- function(z_mean,
                                       z_log_var,
                                       kl_weight,
                                       rec_dim,
                                       y_true,
                                       y_pred){ #TODO: delete this function
  squared <- keras::k_square(z_mean)
  exponential <- keras::k_exp(z_log_var)
  value <- squared + exponential - 1 - z_log_var
  kl_loss <- 0.5 * keras::k_sum(value,
                                axis = -1L)
  rec_loss <- rec_dim * keras::loss_binary_crossentropy(y_true, y_pred)
  vae_loss <- keras::k_mean(kl_weight * kl_loss + rec_loss)
  vae_loss
}

#' This function is deprecated and will be removed in future versions. A custom loss function for a VAE learning a multivariate normal distribution with a full covariance matrix for use with custom training
#'
#' @param z_mean a tensor (layer) representing the mean in the VAE
#' @param z_log_cholesky a tensor (layer) reshaped into a lower triangular matrix, representing the log cholesky matrix in the VAE
#' @param inv_skill_cov a constant tensor matrix of the inverse of the covariance matrix being learned
#' @param det_skill_cov a constant tensor scalar representing the determinant of the covariance matrix being learned
#' @param skill_mean a constant tensor vector representing the means of the latent skills being learned
#' @param kl_weight weight for the KL divergence term
#' @param rec_dim the number of nodes in the input/output of the VAE
#' @param y_true the input to the VAE
#' @param y_pred the output of the VAE
experimental_full_cov_loss <- function(z_mean,
                                       z_log_cholesky,
                                       inv_skill_cov,
                                       det_skill_cov,
                                       skill_mean,
                                       kl_weight,
                                       rec_dim,
                                       y_true,
                                       y_pred){ #TODO: delete this function
  z_cholesky <- tensorflow::tf$linalg$expm(z_log_cholesky)
  z_cov_matrix <- tensorflow::tf$matmul(z_cholesky, tensorflow::tf$transpose(z_cholesky, c(0L, 2L, 1L)))
  num_skills <- keras::k_int_shape(skill_mean)[[2]]
  diff <- tensorflow::tf$reshape(z_mean - skill_mean, c(-1L, num_skills, 1L))
  temp <- keras::k_dot(tensorflow::tf$transpose(diff, c(0L, 2L, 1L)), inv_skill_cov)
  kl_loss <- 0.5 * (tensorflow::tf$linalg$trace(tensorflow::tf$transpose(
    keras::k_dot(inv_skill_cov, z_cov_matrix), c(1L,0L,2L))) +
    keras::k_reshape(tensorflow::tf$matmul(temp, diff), c(-1)) -
    tensorflow::tf$constant(num_skills, dtype = 'float32') +
    log(det_skill_cov / tensorflow::tf$linalg$det(z_cov_matrix)))
  rec_loss <- rec_dim * keras::loss_binary_crossentropy(y_true, y_pred)
  vae_loss <- keras::k_mean(kl_weight * kl_loss + rec_loss)
  vae_loss
}
