#' Get trainable variables from the decoder, which serve as item parameter estimates.
#'
#' @param decoder a trained keras model; can either be the decoder or vae returned from \code{build_vae_independent()} or \code{build_vae_correlated}
#' @param model_type either 1 or 2, specifying a 1 parameter (1PL) or 2 parameter (2PL) model; if 1PL, then only the difficulty parameter estimates (output layer bias) will be returned; if 2PL, then the discrimination parameter estimates (output layer weights) will also be returned
#' @return a list which contains item parameter estimates; the length of this list is equal to model_type - the first entry in the list holds the difficulty parameter estimates, and the second entry (if 2PL) contains discrimination parameter estimates
#' @export
#' @examples
#' \donttest{
#' Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
#' models <- build_vae_independent(4, 2, Q, model_type = 2)
#' decoder <- models[[2]]
#' item_parameter_estimates <- get_item_parameter_estimates(decoder, model_type = 2)
#' difficulty_est <- item_parameter_estimates[[1]]
#' discrimination_est <- item_parameter_estimates[[2]]
#' }
get_item_parameter_estimates <- function(decoder, model_type = 2){
  all_decoder_weights <- keras::get_weights(decoder)
  weights_length <- length(all_decoder_weights)
  estimates <- c()
  if (model_type == 1){
    estimates[1] <- all_decoder_weights[weights_length]
  } else{
    estimates[1] <- all_decoder_weights[weights_length]
    estimates[2] <- all_decoder_weights[weights_length - 1]
  }
  estimates
}

#' Feed forward response sets through the encoder, which outputs student ability estimates
#'
#' @param encoder a trained keras model; should be the encoder returned from either \code{build_vae_independent()} or \code{build_vae_correlated}
#' @param responses a \code{num_students} by \code{num_items} matrix of binary responses, as used in training
#' @return a list where the first entry contains student ability estimates and the second entry holds the variance (or covariance matrix) of those estimates
#' @export
#' @examples
#' \donttest{
#' data <- matrix(c(1,1,0,0,1,0,1,1,0,1,1,0), nrow = 3, ncol = 4)
#' Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
#' models <- build_vae_independent(4, 2, Q, model_type = 2)
#' encoder <- models[[1]]
#' ability_parameter_estimates_variances <- get_ability_parameter_estimates(encoder, data)
#' student_ability_est <- ability_parameter_estimates_variances[[1]]
#' }
get_ability_parameter_estimates <- function(encoder, responses){
  encoded_responses <- encoder(responses)
  estimates_variances <- c()
  ability_parameter_estimates <- encoded_responses[[1]]
  ability_parameter_log_variance <- encoded_responses[[2]]
  if (ability_parameter_estimates$shape == ability_parameter_log_variance$shape){
    estimates_variances[[1]] <- ability_parameter_estimates$numpy()
    estimates_variances[[2]] <- exp(ability_parameter_log_variance$numpy())
  } else{
    b <- tfprobability::tfb_fill_triangular(upper=FALSE)
    log_cholesky <- b$forward(ability_parameter_log_variance)
    cholesky <- tensorflow::tf$linalg$expm(log_cholesky)
    cov_matrices <- tensorflow::tf$matmul(cholesky, tensorflow::tf$transpose(cholesky, c(0L, 2L, 1L)))
    estimates_variances[[1]] <- ability_parameter_estimates$numpy()
    estimates_variances[[2]] <- cov_matrices
  }
  estimates_variances
}
