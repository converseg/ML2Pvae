#' Build the encoder for a VAE
#'
#' @param input_size an integer representing the number of items
#' @param layers a list of integers giving the size of each hidden layer
#' @param activations a list of strings, the same length as layers
#' @return two tensors: the input layer to the VAE and the last hidden layer of the encoder
build_hidden_encoder <- function(input_size,
                          layers,
                          activations = rep('sigmoid', length(layers))){
  input <- keras::layer_input(shape = c(input_size), name = 'input')
  h <- input
  if (length(layers) > 0){
    for (layer in 1:length(layers)){
      h <- keras::layer_dense(h,
                              units = layers[layer],
                              activation = activations[layer],
                              name = paste('hidden_', layer, sep = ''))
    }
  }
  list(input, h)
}
