#' Create a wrapper function to fit models.
#'
#' @param model the model to be trained
#' @param train_data training data
#' @param validation_split split percent to use as validation data
#' @param shuffle whether or not to shuffle data
#' @param num_epochs number of epochs to train for
#' @param verbose verbosity levels
#' @param batch_size batch size for SGD
#' @export
train_model <- function(model,
                              train_data,
                              validation_split = 0.15,
                              shuffle = FALSE,
                              num_epochs = 10,
                              verbose = 1,
                              batch_size= 1){
  history <- keras::fit(model, train_data, train_data,
                              epochs = num_epochs,
                              shuffle = shuffle,
                              verbose = verbose,
                              batch_size = batch_size)
  history
}

