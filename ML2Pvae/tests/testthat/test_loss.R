context("KL loss")

test_that("standard normal KL divergence is computed correctly", {
  m <- c(0.25, -0.5)
  lv <- c(-0.2, 0.1)
  sess <- tensorflow::tf$Session()
  placeholder <- tensorflow::tf$constant(c(0,0,0))
  kl_loss <- vae_loss_standard_normal(m, lv, 1, 3)(placeholder, placeholder)
  kl_value <- sess$run(kl_loss)
  sess$close()
  expect_equal(kl_value, 0.168201, tolerance = 1e-5)
})

test_that("full covariance KL divergence is computed correctly", {
  sess <- tensorflow::tf$Session()
  target_m <- tensorflow::tf$constant(c(1,2), shape = c(1,2), dtype = 'float32')
  target_cov <- tensorflow::tf$constant(matrix(c(1, 0.25, 0.25, 1.5), nrow = 2, ncol = 2),
                                        shape = c(2,2), dtype = 'float32')
  target_inv <- tensorflow::tf$linalg$inv(target_cov)
  target_det <- tensorflow::tf$linalg$det(target_cov)
  sample_m <- tensorflow::tf$constant(c(.9, 2.3), shape = c(1,1,2), dtype = 'float32')
  sample_log_chol <- tensorflow::tf$constant(matrix(c(-0.0527, 0.5682, 0, -0.0979), nrow = 2, ncol = 2),
                                             shape = c(1,2,2), dtype = 'float32')
  placeholder <- tensorflow::tf$constant(c(0,0,0), dtype = 'float32')
  kl_loss <- vae_loss_normal_full_covariance(sample_m, sample_log_chol,
                                             target_inv, target_det, target_m, 1, 3)(placeholder, placeholder)
  kl_value <- sess$run(kl_loss)
  sess$close()
  expect_equal(kl_value, 0.1390, tolerance = 1e-4)
})