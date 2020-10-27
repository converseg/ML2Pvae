#' Trains a VAE or autoencoder model. This acts as a wrapper for keras::fit().
#'
#' @param model the keras model to be trained; this should be the vae returned from \code{build_vae_standard_normal()} or \code{build_vae_normal_full_covariance}
#' @param train_data training data; this should be a binary \code{num_students} by \code{num_items} matrix of student responses to an assessment
#' @param num_epochs number of epochs to train for
#' @param batch_size batch size for mini-batch stochastic gradient descent; default is 1, detailing pure SGD; if a larger batch size is used (e.g. 32), then a larger number of epochs should be set (e.g. 50)
#' @param validation_split split percentage to use as validation data
#' @param shuffle whether or not to shuffle data
#' @param verbose verbosity levels; 0 = silent; 1 = progress bar and epoch message; 2 = epoch message
#' @return a list containing training history; this holds the loss from each epoch which can be plotted
#' @export
#' @examples
#' \donttest{
#' data <- matrix(c(1,1,0,0,1,0,1,1,0,1,1,0), nrow = 3, ncol = 4)
#' Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
#' models <- build_vae_standard_normal(4, 2, Q)
#' vae <- models[[3]]
#' history <- train_model(vae, data, num_epochs = 3, validation_split = 0, verbose = 0)
#' plot(history)
#' }
train_model <- function(model,
                        train_data,
                        validation_split = 0.15,
                        num_epochs = 10,
                        batch_size = 1,
                        shuffle = FALSE,
                        verbose = 1){
  history <- keras::fit(model, train_data, train_data,
                              epochs = num_epochs,
                              shuffle = shuffle,
                              verbose = verbose,
                              batch_size = batch_size)

  history[2]$metrics$loss
}

