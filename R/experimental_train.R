#' Fit standard normal models with gradient tape rather than keras
#'
#' @param encoder a compiled keras model
#' @param decoder a compiled keras model
#' @param vae a compiled keras model
#' @param train_data the data to fit the model on
#' @param num_epochs how long to train for
#' @param learning_rate the learning rate used for the adam optimizer
#' @param kl_weight weight for KL Divergence
#' @param verbose level of console output
#' @export
fit_standard_model <- function(encoder,
                      decoder,
                      vae,
                      train_data,
                      num_epochs = 10,
                      learning_rate = 0.001,
                      kl_weight = 1,
                      verbose = 1){
  num_train <- nrow(train_data)
  num_items <- ncol(train_data)
  optimizer <- keras::optimizer_adam(lr = learning_rate)
  loss_function <- experimental_standard_loss
  #TODO: figure out how to show a metric
  # loss_metric <- keras::metric_mean_squared_error
  for(epoch in 1:num_epochs){
    if(verbose >= 1){cat('Epoch', epoch, '/', num_epochs, '\n')}
    if(verbose == 1){
      pb <- progress::progress_bar$new(
        format = " [:bar] Elapsed: :elapsed ETA: :eta",
        total = num_train,
        width = 80,
        clear = FALSE)
      pb$tick(0)
    }
    summed_loss <- 0
    for(step in 1:num_train){#TODO: batch size
      if(verbose == 1){pb$tick()}
      responses <- train_data[step,]
      with(tape <- tensorflow::tf$GradientTape(persistent = TRUE),{
        tape$watch(vae$trainable_weights)
        #TODO: separate function for calc loss??
        outputs <- encoder(t(responses))
        z_mean <- outputs[[1]]
        z_log_var <- outputs[[2]]
        z_sample <- outputs[[3]]
        y_pred <- decoder(z_sample)
        loss <- loss_function(z_mean, z_log_var, kl_weight, num_items, responses, y_pred)
      })
      summed_loss <- summed_loss + tensorflow::tf$keras$backend$get_value(loss)
      grads <- tape$gradient(loss, vae$trainable_variables)
      optimizer$apply_gradients(Map(c, grads, vae$trainable_variables))
      #TODO: figure out metric
      # loss_metric(loss)
    }
    if(verbose >= 1){cat(' Loss:', summed_loss/num_train, '\n')}
  }
}

#' Fit full covariacne models with gradient tape rather than keras
#'
#' @param encoder a compiled keras model
#' @param decoder a compiled keras model
#' @param vae a compiled keras model
#' @param mean_vector the mean values of latent skills
#' @param cov_matrix the covariance matrix of latent skills
#' @param train_data the data to fit the model on
#' @param num_epochs how long to train for
#' @param learning_rate the learning rate used for the adam optimizer
#' @param kl_weight weight for KL Divergence
#' @param verbose level of console output
#' @export
fit_full_cov_model <- function(encoder,
                               decoder,
                               vae,
                               mean_vector,
                               cov_matrix,
                               train_data,
                               num_epochs = 10,
                               learning_rate = 0.001,
                               kl_weight = 1,
                               verbose = 1){
  num_train <- nrow(train_data)
  num_skills <- length(mean_vector)
  num_items <- ncol(train_data)
  det_skill_cov <- tensorflow::tf$constant(det(cov_matrix), dtype = 'float32')
  inv_skill_cov <- tensorflow::tf$constant(solve(cov_matrix), dtype = 'float32')
  skill_mean <- tensorflow::tf$constant(mean_vector, shape = c(1L, as.integer(num_skills)), dtype = 'float32')
  optimizer <- keras::optimizer_adam(lr = learning_rate)
  loss_function <- experimental_full_cov_loss
  b <- tfprobability::tfb_fill_triangular(upper=FALSE)
  # loss_metric <- keras::metric_mean_squared_error
  for(epoch in 1:num_epochs){
    if(verbose >= 1){cat('Epoch', epoch, '/', num_epochs, '\n')}
    if(verbose == 1){
      pb <- progress::progress_bar$new(
        format = " [:bar] Elapsed: :elapsed ETA: :eta",
        total = num_train,
        width = 80,
        clear = FALSE)
      pb$tick(0)
    }
    summed_loss <- 0
    for(step in 1:num_train){#TODO: batch size
      if(verbose == 1){pb$tick()}
      responses <- train_data[step,]
      with(tape <- tensorflow::tf$GradientTape(persistent = TRUE),{
        tape$watch(vae$trainable_weights)
        #TODO: separate function for calc loss??
        outputs <- encoder(t(responses))
        z_mean <- outputs[[1]]
        z_log_chol <- outputs[[2]]
        z_sample <- outputs[[3]]
        y_pred <- decoder(z_sample)
        f <- b$forward(z_log_chol)
        loss <- loss_function(z_mean, f, inv_skill_cov, det_skill_cov, skill_mean,
                              kl_weight, num_items, responses, y_pred)
      })
      summed_loss <- summed_loss + tensorflow::tf$keras$backend$get_value(loss)
      grads <- tape$gradient(loss, vae$trainable_variables)
      optimizer$apply_gradients(Map(c, grads, vae$trainable_variables))
      # loss_metric(loss)
    }
    if(verbose >= 1){cat(' Loss:', summed_loss/num_train, '\n')}
  }
}