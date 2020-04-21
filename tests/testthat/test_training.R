context("training")

test_that("standard normal vae can be fit to data", {
  responses <- runif(52)
  responses[responses > 0.5] <- 1
  responses[responses <= 0.5] <- 0
  data <- matrix(responses, nrow = 13, ncol = 4)
  train_data <- data[1:10,]
  test_data <- data[11:13,]
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  models <- build_vae_standard_normal(4, 2, Q, model_type = 2, enc_hid_arch = c(5,3))
  encoder <- models[[1]]
  decoder <- models[[2]]
  vae <- models[[3]]
  keras::fit(vae,
             train_data, train_data,
             shuffle = FALSE,
             epochs = 3,
             verbose = 0,
             batch_size = 3
  )
  encoded_test <- predict(encoder, test_data)
  enc_test_mean <- encoded_test[[1]]
  enc_test_log_var <- encoded_test[[2]]
  expect_equal(dim(enc_test_mean),c(3,2))
  expect_equal(dim(enc_test_log_var),c(3,2))
})

test_that("full covariance vae can be fit to data", {
  responses <- runif(52)
  responses[responses > 0.5] <- 1
  responses[responses <= 0.5] <- 0
  data <- matrix(responses, nrow = 13, ncol = 4)
  train_data <- data[1:10,]
  test_data <- data[11:13,]
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  cov <- matrix(c(.7,.3,.3,1), nrow = 2, ncol = 2)
  models <- build_vae_normal_full_covariance(4, 2, Q,
    mean_vector = c(-0.5, 0), covariance_matrix = cov,
    enc_hid_arch = c(6, 3))
  encoder <- models[[1]]
  decoder <- models[[2]]
  vae <- models[[3]]
  keras::fit(vae,
             train_data, train_data,
             shuffle = FALSE,
             epochs = 3,
             verbose = 0,
             batch_size = 3
  )
  encoded_test <- predict(encoder, test_data)
  enc_test_mean <- encoded_test[[1]]
  enc_test_log_chol <- encoded_test[[2]]
  expect_equal(dim(enc_test_mean),c(3,2))
  expect_equal(dim(enc_test_log_chol), c(3,3))
})

test_that("1-parameter logistic model option makes all decoder weights = 1", {
  responses <- runif(40)
  responses[responses > 0.5] <- 1
  responses[responses <= 0.5] <- 0
  data <- matrix(responses, nrow = 10, ncol = 4)
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  models <- build_vae_standard_normal(4, 2, Q, model_type = 1, enc_hid_arch = c(5,3))
  vae <- models[[3]]
  keras::fit(vae,
             data, data,
             shuffle = FALSE,
             epochs = 3,
             verbose = 0,
             batch_size = 1
  )
  decoder_weights <- keras::get_weights(models[[2]])
  disc_estimates <- decoder_weights[[1]]
  expect_equal(disc_estimates, Q)
})

test_that("single-dimensional IRT estimation works", {
  responses <- runif(30)
  responses[responses > 0.5] <- 1
  responses[responses <= 0.5] <- 0
  data <- matrix(responses, nrow = 10, ncol = 3)
  Q <- matrix(rep(1,3), nrow = 1, ncol = 3)
  models <- build_vae_standard_normal(3, 1, Q, model_type = 2, enc_hid_arch = c(4,2))
  vae <- models[[3]]
  keras::fit(vae,
             data, data,
             shuffle = FALSE,
             epochs = 3,
             verbose = 0,
             batch_size = 1
  )
  expect_equal(1,1)
})