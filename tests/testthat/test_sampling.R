context("sampling")

test_that("standard normal samples are generated correctly", {
  skip_on_cran()
  data <- matrix(c(1, 1, -0.5, -0.5, 0, 0, .3, .3), nrow = 2, ncol = 4)
  data <- tensorflow::tf$constant(data, dtype = 'float32')
  samples <- sampling_standard_normal(data)
  result <- tensorflow::tf$get_static_value(samples)
  expect_equal(dim(result), c(2,2))
  expect_false(identical(result[1,], result[2,])) #two random vectors should not be the same
})

test_that("full covariance samples are generated correctly", {
  skip_on_cran()
  data <- matrix(c(1, 1, 1, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1), nrow = 3, ncol = 5)
  data <- tensorflow::tf$constant(data, dtype = 'float32')
  samples <- sampling_normal_full_covariance(data)
  result <- tensorflow::tf$get_static_value(samples)
  expect_equal(dim(result), c(3,2))
  expect_false(identical(result[1,], result[2,])) #two random vectors should not be the same
  expect_false(identical(result[1,], result[3,]))
})
