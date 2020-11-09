#' Build a VAE that fits to a normal, full covariance N(m,S) latent distribution
#'
#' @param num_items an integer giving the number of items on the assessment; also the number of nodes in the input/output layers of the VAE
#' @param num_skills an integer giving the number of skills being evaluated; also the dimensionality of the distribution learned by the VAE
#' @param Q_matrix a binary, \code{num_skills} by \code{num_items} matrix relating the assessment items with skills
#' @param mean_vector a vector of length \code{num_skills} specifying the mean of each latent trait; the default of \code{rep(0, num_skills)} should almost always be used
#' @param covariance_matrix a symmetric, positive definite, \code{num_skills} by \code{num_skills} matrix giving the covariance of the latent traits
#' @param model_type either 1 or 2, specifying a 1 parameter (1PL) or 2 parameter (2PL) model; if 1PL, then all decoder weights are fixed to be equal to one
#' @param enc_hid_arch a vector detailing the size of hidden layers in the encoder; the number of hidden layers is determined by the length of this vector
#' @param hid_enc_activations a vector specifying the activation function in each hidden layer in the encoder; must be the same length as \code{enc_hid_arch}
#' @param output_activation a string specifying the activation function in the output of the decoder; the ML2P model always used 'sigmoid'
#' @param kl_weight an optional weight for the KL divergence term in the loss function
#' @param l1_penalty the weight on a L1-regularization term on the decoder weights with default 0 (i.e. no penalty) - an L1 penalty can be useful if the Q-matrix is unknown
#' @param learning_rate an optional parameter for the adam optimizer
#' @return returns three keras models: the encoder, decoder, and vae
#' @export
#' @examples
#' \donttest{
#' Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
#' cov <- matrix(c(.7,.3,.3,1), nrow = 2, ncol = 2)
#' models <- build_vae_correlated(4, 2, Q,
#'           mean_vector = c(-0.5, 0), covariance_matrix = cov,
#'           enc_hid_arch = c(6, 3), hid_enc_activation = c('sigmoid', 'relu'),
#'           output_activation = 'tanh',
#'           kl_weight = 0.1)
#' vae <- models[[3]]
#' }
build_vae_correlated <- function(num_items,
                                 num_skills,
                                 Q_matrix,
                                 mean_vector = rep(0, num_skills),
                                 covariance_matrix = diag(num_skills),
                                 model_type = 2,
                                 enc_hid_arch = c(ceiling((num_items + num_skills) / 2)),
                                 hid_enc_activations = rep('sigmoid', length(enc_hid_arch)),
                                 output_activation = 'sigmoid',
                                 kl_weight = 1,
                                 l1_penalty = 0,
                                 learning_rate = 0.001){
  validate_inputs(num_items,
                  num_skills,
                  Q_matrix,
                  model_type,
                  mean_vector,
                  covariance_matrix,
                  enc_hid_arch,
                  hid_enc_activations,
                  output_activation,
                  kl_weight,
                  learning_rate)
  if (model_type == 1){
    weight_constraint <- q_1pl_constraint
  } else if (model_type == 2){
    weight_constraint <- q_constraint
  }
  det_skill_cov <- tensorflow::tf$constant(det(covariance_matrix), dtype = 'float32')
  inv_skill_cov <- tensorflow::tf$constant(solve(covariance_matrix), dtype = 'float32')
  skill_mean <- tensorflow::tf$constant(mean_vector, shape = c(1L, as.integer(num_skills)), dtype = 'float32')

  encoder_layers <- build_hidden_encoder(num_items, enc_hid_arch, hid_enc_activations)
  input <- encoder_layers[[1]]
  h <- encoder_layers[[2]]
  z_mean <- keras::layer_dense(h, units = num_skills, activation = 'linear', name = 'z_mean')
  z_log_cholesky <- keras::layer_dense(h, units = num_skills * (num_skills+1) / 2, activation = 'linear', name = 'z_log_cholesky')
  z <- keras::layer_lambda(keras::layer_concatenate(list(z_mean, z_log_cholesky), name = 'z'),
                           sampling_correlated)
  encoder <- keras::keras_model(input, c(z_mean, z_log_cholesky, z))

  latent_inputs <- keras::layer_input(shape = num_skills, name = 'latent_inputs')
  out <- keras::layer_dense(latent_inputs,
                            units = num_items,
                            activation = 'sigmoid',
                            kernel_constraint = weight_constraint(Q_matrix),
                            kernel_regularizer = keras::regularizer_l1(l = l1_penalty),
                            name = 'vae_out')
  decoder <- keras::keras_model(latent_inputs, out)
  output <- decoder(encoder(input)[3])

  vae <- keras::keras_model(input, output)
  vae_loss <- vae_loss_correlated(encoder,
                                  inv_skill_cov,
                                  det_skill_cov,
                                  skill_mean,
                                  kl_weight,
                                  num_items)
  keras::compile(vae,
                 optimizer = keras::optimizer_adam(lr = learning_rate),
                 loss = vae_loss)
  list(encoder, decoder, vae)
}
