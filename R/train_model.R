#' Trains a VAE or autoencoder model. This acts as a wrapper for keras::fit().
#'
#' @param model the model to be trained
#' @param train_data training data
#' @param num_epochs number of epochs to train for
#' @param batch_size batch size for SGD
#' @param validation_split split percent to use as validation data
#' @param shuffle whether or not to shuffle data
#' @param verbose verbosity levels
#' @return returns a list containing training history, which contains the loss from each epoch
#' @export
#' @examples
#' data <- matrix(c(1,1,0,0,1,0,1,1,0,1,1,0), nrow = 3, ncol = 4)
#' Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
#' models <- build_vae_standard_normal(4, 2, Q)
#' vae <- models[[3]]
#' history <- train_model(vae, data, num_epochs = 3, validation_split = 0, verbose = 0)
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
