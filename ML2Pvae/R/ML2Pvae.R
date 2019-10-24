#' ML2Pvae: A package for creating a VAE whose decoder recovers the parameters of the ML2P model.
#' The encoder can be used to predict the latent skills based on assessment scores.
#' 
#' The ML2Pvae package includes functions which build a VAE with the desired architecture, and
#' fits the latent skills to either a standard normal (independent) distrubution,
#' or a multivariate normal distribution with a full covariance matrix. Based on the work 
#' "Interpretable Variational Autoencdoers for Cognitive Models"
#' by Curi, M., Converse, G., Hajewski, J., and Oliveira, S. 
#' Found in International Joint Conference on Neural Networks, 2019.
#'
#' @docType package
#' @name ML2Pvae
NULL