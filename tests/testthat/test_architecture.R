context("architecture")

test_that("standard normal encoder architecture is as specified", {
  skip_on_cran()
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  models <- build_vae_standard_normal(4, 2, Q,
           enc_hid_arch = c(6, 3))
  w <- models[[1]]$get_weights()
  expect_equal(c(dim(w[[3]])[1], dim(w[[3]])[2]) , c(6,3))
})

test_that("full covariance encoder architecture is as specified", {
  skip_on_cran()
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  cov <- matrix(c(.7,.3,.3,1), nrow = 2, ncol = 2)
  models <- build_vae_normal_full_covariance(4, 2, Q,
    mean_vector = c(-0.5, 0), covariance_matrix = cov,
    enc_hid_arch = c(6, 3))
  w <- models[[1]]$get_weights()
  expect_equal(c(dim(w[[3]])[1], dim(w[[3]])[2]) , c(6,3))
})
