context("input")

test_that("detect invalid Q-matrix", {
  #must be binary and correct dimensions
  num_skills <- 2
  num_items <- 4
  Q1 <- matrix(c(1,0,1,1,0,0,1,1), nrow = 4, ncol = 2)
  Q2 <- matrix(c(0.5,-3,1,0,0.1,-1,1,3.14), nrow = 2, ncol = 4)
  expect_error(validate_inputs(num_items, num_skills, Q1),
               'Invalid dimensions for Q_matrix - must be num_skills by num_items.')
  expect_error(validate_inputs(num_items, num_skills, Q2),
               'Entries in Q_matrix must be either 1 or 0.')
})

test_that("detect invalid model type", {
  num_skills <- 2
  num_items <- 4
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  expect_error(validate_inputs(num_items, num_skills, Q, model_type=3),
               'Invalid input for \'model_type\'. Use either 1 for 1PL model, or 2 for 2PL model.')
})

test_that("detect invalid mean an covariance matrix dimensions", {
  num_skills <- 2
  num_items <- 4
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  cov <- diag(4)
  expect_error(validate_inputs(num_items, num_skills, Q, mean_vector = c(0,0,0)),
               'Length of mean_vector must be equal to num_skills.')
  expect_error(validate_inputs(num_items, num_skills, Q, covariance_matrix = cov),
               'Dimensions of covariance_matrix must be num_skills by num_skills.')
})

test_that("detect covariance matrix that is not positive definite", {
  num_skills <- 2
  num_items <- 4
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  cov <- matrix(c(1,1.5,1.5,1.25), nrow = 2, ncol = 2)
  expect_error(validate_inputs(num_items, num_skills, Q, covariance_matrix = cov),
               'The covariance_matrix must be positive definite.')
})

test_that("detect non-numeric architecture", {
  num_skills <- 2
  num_items <- 4
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  enc_arch1 <- c("one", 2, "3")
  enc_arch2 <- c(5, 3, -1)
  expect_error(validate_inputs(num_items, num_skills, Q, enc_hid_arch = enc_arch1),
               'The enc_hid_arch must be a numeric vector.')
  expect_error(validate_inputs(num_items, num_skills, Q, enc_hid_arch = enc_arch2),
               'The number of nodes in each hidden layer must be greater than or equal to 1.')
  
})

test_that("detect invalid encoder architecture", {
  num_skills <- 2
  num_items <- 4
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  enc_arch <- c(6,3,1)
  activations <- c('sigmoid', 'relu')
  expect_error(validate_inputs(num_items, num_skills, Q, enc_hid_arch = enc_arch, hid_enc_activations = activations),
               'The enc_hid_arch and hid_enc_activations must be the same length.')
})

test_that("detect invalid activation functions", {
  num_skills <- 2
  num_items <- 4
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  enc_arch <- c(4,3)
  activations <- c('sigmoid', 'custom_activation')
  expect_error(validate_inputs(num_items, num_skills, Q, enc_hid_arch = enc_arch, hid_enc_activations = activations),
               'Strings in hid_enc_activations and output_activation must be valid activation functions supported by Tensorflow.')
})

test_that("detect negative weight for KL divergence", {
  num_skills <- 2
  num_items <- 4
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  kl <- -0.1
  expect_error(validate_inputs(num_items, num_skills, Q, kl_weight =kl),
               'The kl_weight must be greater than or equal to 0.')
})

test_that("valid inputs do not throw errors", {
  num_skills <- 2
  num_items <- 4
  Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
  m_type <- 1
  mean <- c(-1,0)
  cov <- matrix(c(1,.3,.3,1), nrow = 2, ncol = 2)
  enc_arch <- c(4,3,3)
  activations <- c('sigmoid', 'relu', 'sigmoid')
  out_activation <- 'hard_sigmoid'
  kl <- 1.5
  expect_null(validate_inputs(num_items, num_skills, Q,
                                 model_type = m_type,
                                 mean_vector = mean,
                                 covariance_matrix = cov,
                                 enc_hid_arch = enc_arch,
                                 hid_enc_activations = activations,
                                 output_activation = out_activation,
                                 kl_weight = kl))
})