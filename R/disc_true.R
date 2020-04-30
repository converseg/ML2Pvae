#' Simulated discrimination parameters
#' 
#' Difficulty parameters for an exam of 15 items assessing 3 latent abilities.
#' 
#' @source Each entry is sampled uniformly from \code{[0.25,1.5]}.
#' If an entry in \code{q_matrix.rda} is 0, then so is the corresponding entry in \code{disc_true.rda}.
#' @format A data frame with 3 rows and 15 columns. Entry \code{[k,i]} represents the discrimination 
#' parameter between item \code{i} and ability \code{k}.
#' 
"disc_true"