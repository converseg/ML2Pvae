context("training")

test_that("standard normal vae can be fit to data", {
  skip_on_cran()
  responses <- runif(52)
  responses[responses > 0.5] <- 1
  responses[responses <= 0.5] <- 0
  data <- matrix(responses, nrow = 13, ncol = 4)
  train_data <- data[1:10,]
  test_data <- data[11:13,]
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  models <- build_vae_independent(4, 2, Q, model_type = 2, enc_hid_arch = c(5,3))
  encoder <- models[[1]]
  decoder <- models[[2]]
  vae <- models[[3]]
  history <- train_model(vae, train_data, validation_split = 0.1, num_epochs = 3, verbose=0)
  estimates <- get_ability_parameter_estimates(encoder, test_data)
  enc_test_mean <- estimates[[1]]
  enc_test_var <- estimates[[2]]
  expect_equal(dim(enc_test_mean),c(3,2))
  expect_equal(dim(enc_test_var),c(3,2))
})

test_that("full covariance vae can be fit to data", {
  skip_on_cran()
  responses <- runif(52)
  responses[responses > 0.5] <- 1
  responses[responses <= 0.5] <- 0
  data <- matrix(responses, nrow = 13, ncol = 4)
  train_data <- data[1:10,]
  test_data <- data[11:13,]
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  cov <- matrix(c(.7,.3,.3,1), nrow = 2, ncol = 2)
  mean_vector <- c(-0.5, 0)
  models <- build_vae_correlated(4, 2, Q,
    mean_vector = mean_vector, covariance_matrix = cov,
    enc_hid_arch = c(6, 3))
  encoder <- models[[1]]
  decoder <- models[[2]]
  vae <- models[[3]]
  history <- train_model(vae, train_data, validation_split = 0.1, batch_size = 3, num_epochs = 3, verbose=0)
  estimates <- get_ability_parameter_estimates(encoder, test_data)
  enc_test_mean <- estimates[[1]]
  enc_test_cov <- estimates[[2]]
  expect_equal(dim(enc_test_mean),c(3,2))
  expect_equal(dim(enc_test_cov), c(3,2,2))
})

test_that("1-parameter logistic model option makes all decoder weights = 1", {
  skip_on_cran()
  responses <- runif(40)
  responses[responses > 0.5] <- 1
  responses[responses <= 0.5] <- 0
  data <- matrix(responses, nrow = 10, ncol = 4)
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  models <- build_vae_independent(4, 2, Q, model_type = 1, enc_hid_arch = c(5,3))
  encoder <- models[[1]]
  decoder <- models[[2]]
  vae <- models[[3]]
  history <- train_model(vae, data, num_epochs = 3, verbose=0)
  decoder_weights <- keras::get_weights(decoder)
  disc_estimates <- decoder_weights[[1]]
  expect_equal(disc_estimates, Q)
})

test_that("single-dimensional IRT estimation works", {
  skip_on_cran()
  responses <- runif(30)
  responses[responses > 0.5] <- 1
  responses[responses <= 0.5] <- 0
  data <- matrix(responses, nrow = 10, ncol = 3)
  Q <- matrix(rep(1,3), nrow = 1, ncol = 3)
  models <- build_vae_independent(3, 1, Q, model_type = 2, enc_hid_arch = c(4,2))
  encoder <- models[[1]]
  decoder <- models[[2]]
  vae <- models[[3]]
  history <- train_model(vae, data, num_epochs = 3, verbose=0)
  item_parameter_estimates <- get_item_parameter_estimates(decoder, model_type = 2)
  expect_equal(1,1)
})
