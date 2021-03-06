% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/q_matrix.R
\docType{data}
\name{q_matrix}
\alias{q_matrix}
\title{Simulated Q-matrix}
\format{
A data frame with 3 rows and 30 columns. If entry \code{[k,i] = 1},
then item \code{i} requires skill \code{k}.
}
\source{
Generated by sampling each entry from \code{Bernoulli(0.35)}, but ensures
each item assess at least one latent ability
}
\usage{
q_matrix
}
\description{
The Q-matrix determines the relation between items and abilities.
}
\keyword{datasets}
