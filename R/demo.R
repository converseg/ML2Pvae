#' A demo on how to construct, train, and evaluate an ML2P-VAE model
#'
load('.\\data\\responses.rda')
load('.\\data\\q_matrix.rda')
load('.\\data\\correlation_matrix.rda')
data <- as.matrix(responses)
Q <- as.matrix(q_matrix)
cov <- as.matrix(correlation_matrix)

# Model parameters
num_items <- as.double(dim(Q)[2])
num_skills <- as.double(dim(Q)[1])
num_students <- dim(data)[1]
means <- rep(0,num_skills)
enc_arch <- c(8L,8L,4L)
enc_act <- c('tanh', 'tanh', 'tanh')
out_act <- 'sigmoid'
kl <- 1.0

# Construct ML2P-VAE model
models <- build_vae_normal_full_covariance(num_items,
                                           num_skills,
                                           Q,
                                           model_type = 2,
                                           mean_vector = means,
                                           covariance_matrix = cov,
                                           enc_hid_arch = enc_arch,
                                           hid_enc_activations = enc_act,
                                           output_activation = out_act,
                                           kl_weight = kl)
encoder <- models[[1]]
decoder <- models[[2]]
vae <- models[[3]]

# Training parameters
num_train <- floor(0.8 * num_students)
num_test <- num_students - num_train
data_train <- data[1:num_train,]
data_test <- data[(num_train+1):num_students,]
num_epochs <- 50
batch_size <- 1

# Train ML2P-VAE model
keras::fit(object = vae,
           x = data_train,
           y = data_train,
           batch_size = batch_size,
           epochs = num_epochs,
           shuffle = FALSE,
           verbose = 1)

# Get parameter estimates
decoder_params <- keras::get_weights(decoder)
disc_est <- decoder_params[[1]]
diff_est <- decoder_params[[2]]
test_theta_est <- predict(encoder, data_test)[[1]]
all_theta_est <- predict(encoder, data)[[1]]

# Load in true values
load('.\\data\\disc_true.rda')
load('.\\data\\diff_true.rda')
load('.\\data\\theta_true.rda')
disc_true <- as.matrix(disc_true)
diff_true <- as.matrix(diff_true)
theta_true<- as.matrix(theta_true)

# Evaluate ML2P-VAE model estimates
matplot(t(disc_true), t(disc_est), pch = '*')
plot(diff_true, diff_est)
matplot(theta_true, all_theta_est, pch = '*')
