#'
#' TODO: documentation
#'
#'
#'
validate_inputs <- function(num_items,
                            num_skills,
                            Q_matrix,
                            model_type = 2,
                            mean_vector = rep(0, num_skills),
                            covariance_matrix = diag(num_skills),
                            enc_hid_arch = c(ceiling((num_items + num_skills)/2)),
                            hid_enc_activations = rep('sigmoid', length(enc_hid_arch)),
                            output_activation = 'sigmoid',
                            kl_weight = 1){
  message <- ''
  if (nrow(Q_matrix) != num_skills || ncol(Q_matrix) != num_items){
    message <- paste(message, 'Invalid dimensions for Q_matrix - must be num_skills by num_items.', sep = '\n')
  }

  for (i in 1:num_items*num_skills){
    if (Q_matrix[i] != 1 && Q_matrix[i] != 0){
      message <- paste(message, 'Entries in Q_matrix must be either 1 or 0.', sep = '\n')
      break
    }
  }
  
  if (model_type == 1){
    weight_constraint <- q_1pl_constraint
  } else if (model_type == 2){
    weight_constraint <- q_constraint
  } else{
    message <- paste(message, 'Invalid input for \'model_type\'. Use either 1 for 1PL model, or 2 for 2PL model.', sep = '\n')
  }
  
  #check dimensions of mean and covariance matrix
  if (length(mean_vector) != num_skills){
    message <- paste(message, 'Length of mean_vector must be equal to num_skills.', sep = '\n')
  }
  if (nrow(covariance_matrix) != ncol(covariance_matrix) || nrow(covariance_matrix) != num_skills){
    message <- paste(message, 'Dimensions of covariance_matrix must be num_skills by num_skills.', sep = '\n')
  }
  
  #check covariance matrix is pos def
  m <- tryCatch(chol(covariance_matrix), error = function(err){
    return('The covariance_matrix must be positive definite.')})
  if (identical(m,'The covariance_matrix must be positive definite.')){
    message <- paste(message, m, sep = '\n')
  }
  
  #check for invalid architecture
  if (typeof(enc_hid_arch) != "double" && typeof(enc_hid_arch) != "integer"){
    message <- paste(message, 'The enc_hid_arch must be a numeric vector.', sep = '\n')
  } else if (min(enc_hid_arch) < 1){
    message <- paste(message, 'The number of nodes in each hidden layer must be greater than or equal to 1.', sep = '\n')
  }
  
  #check architecture sizes line up
  if (length(enc_hid_arch) != length(hid_enc_activations)){
    message <- paste(message, 'The enc_hid_arch and hid_enc_activations must be the same length.', sep = '\n')
  }
  
  #check for valid activation functions
  valid_activations <- c('elu',
                         'exponential',
                         'hard_sigmoid',
                         'linear',
                         'relu',
                         'selu',
                         'sigmoid',
                         'softmax',
                         'softplus',
                         'softsign',
                         'tanh')
  given_activations <- append(hid_enc_activations, output_activation)
  for (str in given_activations){
    if (!is.element(str, valid_activations)){
      message <- paste(message, 'Strings in hid_enc_activations and output_activation must be valid activation functions supported by Tensorflow.', sep = '\n')
      break
    }
  }
  
  #make sure KL weight is >=0
  if (kl_weight < 0){
    message <- paste(message, 'The kl_weight must be greater than or equal to 0.', sep = '\n')
  }
  
  # print out error message
  if (message != ''){
    stop(message)
  }
}