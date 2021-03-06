context("KL loss")
test_that("standard normal KL divergence is computed correctly", {
  skip_on_cran()
  m1 <- tensorflow::tf$constant(c(0.25, -0.5), shape = c(1L,2L))
  lv1 <- tensorflow::tf$constant(c(-0.2, 0.1), shape = c(1L,2L))
  m2 <- tensorflow::tf$constant(c(0, -.25), shape = c(1L,2L))
  lv2 <- tensorflow::tf$constant(c(-1, .5), shape = c(1L,2L))
  means <- tensorflow::tf$concat(c(m1,m2), 0L)
  log_vars <- tensorflow::tf$concat(c(lv1, lv2), 0L)
  placeholder <- tensorflow::tf$constant(c(0,0,0))
  dummy_encoder <- function(input){
    c(means, log_vars)
  }
  kl_loss <- vae_loss_independent(dummy_encoder, 1, 3)(placeholder, placeholder)
  kl_value <- tensorflow::tf$get_static_value(kl_loss)
  expect_equal(kl_value, 0.5 * (0.168201 + 0.2895504), tolerance = 1e-5)
})

test_that("full covariance KL divergence is computed correctly", {
  skip_on_cran()
  target_m <- tensorflow::tf$constant(c(1,2), shape = c(1L,2L), dtype = 'float32')
  target_cov <- tensorflow::tf$constant(matrix(c(1, 0.25, 0.25, 1.5), nrow = 2, ncol = 2),
                                        shape = c(2L,2L), dtype = 'float32')
  target_inv <- tensorflow::tf$linalg$inv(target_cov)
  target_det <- tensorflow::tf$linalg$det(target_cov)
  s1_m <- tensorflow::tf$constant(c(.9, 2.3), shape = c(1L,2L), dtype = 'float32')
  s1_log_chol <- tensorflow::tf$constant(c(-0.0979, 0.5682,-0.0527),
                                             shape = c(1L,3L), dtype = 'float32')
  s2_m <- tensorflow::tf$constant(c(-0.3, 0), shape = c(1L,2L), dtype = 'float32')
  s2_log_chol <- tensorflow::tf$constant(c(-0.2554128, 0.4510282, -0.4581454),
                                         shape = c(1L,3L), dtype = 'float32')
  sample_m <- tensorflow::tf$concat(c(s1_m, s2_m), 0L)
  sample_log_chol <- tensorflow::tf$concat(c(s1_log_chol, s2_log_chol), 0L)
  placeholder <- tensorflow::tf$constant(c(0,0,0), dtype = 'float32')
  dummy_encoder <- function(input){
    c(sample_m, sample_log_chol)
  }
  kl_loss <- vae_loss_correlated(dummy_encoder,
                                             target_inv, target_det, target_m, 1, 3)(placeholder, placeholder)
  kl_value <- tensorflow::tf$get_static_value(kl_loss)
  expect_equal(kl_value, 0.5 *(0.1390 + 2.133272), tolerance = 1e-4)
})
