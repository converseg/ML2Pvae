
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

<!-- badges: end -->

# ML2Pvae

This R package allows for constructing, training, and evaluating
ML2P-VAE parameter estimation models in Item Response Theory (IRT).
These methods are based off of the work of Curi et
al. (<https://ieeexplore.ieee.org/document/8852333>). Specific
modifications are made to the VAE architecture which allow for
interpretation of some of the trainable weights/biases, as well as the
values of a hidden layer of the neural network.

## Installation

The ML2Pvae package is hosted on CRAN and can be downloaded by running

    install.packages("ML2Pvae")}

Alternatively, ML2Pvae can be installed from GitHub using the command

    library(devtools)
    install_github("converseg/ML2Pvae")

Note that ML2Pvae requires an installation of Python 3, along with the Python libraries ‘keras’, ‘tensorflow’, and ‘tensorflow-probability’.

## Example

We demonstrate the workflow of this package on a simulated dataset which
consists of responses by 5000 subjects on an assessment with 30 items.
It is assumed that the assessment evaluates 3 latent skills. This is a
vanilla example, and much more detail and explanation is given to the
examples found in the “vignettes” directory.

``` r
library(ML2Pvae)
#> Warning: package 'ML2Pvae' was built under R version 3.6.3

# Load sample data - included in package
data <- as.matrix(responses)
Q <- as.matrix(q_matrix)
```

First set a few hyper-parameters to build the neural network. Note that
`num_items` and `num_skills` are fixed by the data set. The core
function here is `build_vae_independent()`, which constructs a modified
neural network which can be used for parameter estimation.

``` r
# Model parameters
num_items <- as.double(dim(Q)[2])
num_skills <- as.double(dim(Q)[1])
num_students <- dim(data)[1]
enc_arch <- c(16L, 8L)
enc_act <- c('relu', 'tanh')
out_act <- 'sigmoid'
kl <- 1

models <- build_vae_independent(
  num_items,
  num_skills,
  Q,
  model_type = 2,
  enc_hid_arch = enc_arch,
  hid_enc_activation = enc_act,
  output_activation = out_act)
encoder <- models[[1]]
decoder <- models[[2]]
vae <- models[[3]]
```

Next, we set training parameters and fit the model to the data.

``` r
# Training parameters
num_train <- floor(0.8 * num_students)
num_test <- num_students - num_train
data_train <- data[1:num_train,]
data_test <- data[(num_train + 1):num_students,]
num_epochs <- 15
batch_size <- 8

# Train model
history <- train_model(
  vae,
  data_train,
  num_epochs = num_epochs,
  batch_size = batch_size,
  verbose = 1)
```

After the VAE has been trained, we can obtain IRT parameter estimates.
Due to the modifications to the neural architecture, we are able to
interpret the learned weights/biases in the VAE decoder as
discrimination/difficulty parameter estimates. We can also feed forward
student responses through the decoder to obtain estimates for student
abilities.

``` r
# Get IRT parameter estimates 
item_param_estimates <- get_item_parameter_estimates(
  decoder, model_type = 2)
diff_est <- item_param_estimates[[1]]
disc_est <- item_param_estimates[[2]]
test_theta_est <- get_ability_parameter_estimates(
  encoder, data_test)[[1]]
all_theta_est <- get_ability_parameter_estimates(
  encoder, data)[[1]]
```

Though real-world datasets likely won’t have “true” values of the
parameters available, our simulated dataset includes them for reference.
Of course, none of these were used when we trained the model.

``` r
# Load in true values (included in this package)
disc_true <- as.matrix(disc_true)
diff_true <- as.matrix(diff_true)
theta_true<- as.matrix(theta_true)
```

<img src="README_files/figure-gfm/unnamed-chunk-7-1.png" width="100%" />

This was a very basic example, and results can be improved by
fine-tuning the network architecture.

## References

Curi et. al. “Interpretable Variational Autoencoders for Cognitive
Models.” In Proceddings of the International Joint Conference on Neural
Networks (IJCNN),2019.

Converse, Curi, Oliveira. “Autoencoders for Educational Assessment.” In
Proceedings of the Conference on Artifical Intelligence in Education
(AIED), 2019.
