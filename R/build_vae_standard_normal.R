#' Build a VAE that fits to a standard N(0,I) latent distribution with independent latent traits
#'
#' @param num_items the number of items on the assessment; also the number of nodes in the input/output layers of the VAE
#' @param num_skills the number of skills being evaluated; also the size of the distribution learned by the VAE
#' @param Q_matrix a binary, \code{num_skills} by \code{num_items} matrix relating the assessment items with skills
#' @param model_type either 1 or 2, specifying a 1 parameter (1PL) or 2 parameter (2PL) model
#' @param enc_hid_arch a vector detailing the number an size of hidden layers in the encoder
#' @param hid_enc_activations a vector specifying the activation function in each hidden layer in the encoder; must be the same length as \code{enc_hid_arch}
#' @param output_activation a string specifying the activation function in the output of the decoder; the ML2P model alsways used 'sigmoid'
#' @param kl_weight an optional weight for the KL divergence term in the loss function
#' @return Returns three keras models: the encoder, decoder, and vae.
#' @export
#' @examples
#' Q <- matrix(c(1,0,1,1,0,1,1,0), nrow = 2, ncol = 4)
#' models <- build_vae_standard_normal(4, 2, Q,
#'           enc_hid_arch = c(6, 3), hid_enc_activation = c('sigmoid', 'relu'),
#'           output_activation = 'tanh', kl_weight = 0.1)
#' models <- build_vae_standard_normal(4, 2, Q)
#' vae <- models[[3]]
build_vae_standard_normal <- function(num_items,
                                      num_skills,
                                      Q_matrix,
                                      model_type = 2,
                                      enc_hid_arch=c(ceiling((num_items + num_skills)/2)),
                                      hid_enc_activations=rep('sigmoid', length(enc_hid_arch)),
                                      output_activation='sigmoid',
                                      kl_weight=1){#TODO: remove kl weight from building
  validate_inputs(num_items,
                             num_skills,
                             Q_matrix,
                             model_type,
                             rep(0, num_skills),
                             diag(num_skills),
                             enc_hid_arch,
                             hid_enc_activations,
                             output_activation,
                             kl_weight)
  if (model_type == 1){
    weight_constraint <- q_1pl_constraint
  } else if (model_type == 2){
    weight_constraint <- q_constraint
  }
  encoder_layers <- build_hidden_encoder(num_items, enc_hid_arch, hid_enc_activations)
  input <- encoder_layers[[1]]
  h <- encoder_layers[[2]]
  z_mean <- keras::layer_dense(h, units = num_skills, activation = 'linear', name = 'z_mean')
  z_log_var <- keras::layer_dense(h, units = num_skills, activation = 'linear', name = 'z_log_var')
  z <- keras::layer_lambda(keras::layer_concatenate(list(z_mean, z_log_var), name = 'z'),
                           sampling_standard_normal)
  encoder <- keras::keras_model(input, c(z_mean, z_log_var, z))

  latent_inputs <- keras::layer_input(num_skills, name = 'latent_inputs')
  out <- keras::layer_dense(latent_inputs,
                            units = num_items,
                            activation = output_activation,
                            kernel_constraint = weight_constraint(Q_matrix),
                            name = 'vae_out')
  decoder <- keras::keras_model(latent_inputs, out)
  output <- decoder(encoder(input)[3])

  vae <- keras::keras_model(input, output)
  vae_loss <- vae_loss_standard_normal(encoder, kl_weight, num_items)
  keras::compile(vae,
                 optimizer = keras::optimizer_adam(),
                 loss = vae_loss)
  list(encoder, decoder, vae)
}
