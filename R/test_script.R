#' TODO this whole script - mayble load in CDM data
#' This is a script for testing a VAE that fits a standard normal distribution
test_standard_normal <- function(){
  num_items <- 28
  num_skills <- 3

  Q <- matrix(c(
    1, 1, 0,
    0, 1, 0,
    1, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    1, 0, 1,
    0, 1, 0,
    0, 0, 1,
    1, 0, 0,
    1, 0, 1,
    1, 0, 1,
    1, 0, 0,
    1, 0, 0,
    0, 0, 1,
    1, 0, 1,
    0, 1, 1,
    0, 0, 1,
    0, 0, 1,
    1, 0, 1,
    1, 0, 1,
    0, 0, 1,
    0, 1, 0,
    0, 1, 0,
    1, 0, 0,
    0, 0, 1,
    1, 0, 0,
    0, 0, 1), nrow=num_items, ncol=num_skills, byrow=TRUE)
  colnames(Q) <- paste("Dim",c(1:num_skills),sep="")
  rownames(Q) <- paste("Item",c(1:num_items),sep="")
  Q = t(Q)

  models <- build_vae_standard_normal(num_items, num_skills, Q, enc_hid_arch=c(15,10))
  vae <- models[[3]]
  test_fit_model(models, Q, 3)

}

#' TODO documentation
#' This is a script for fitting a normal distribution with a full covariance matrix
test_normal_full_covariance <- function(){
  num_items <- 28
  num_skills <- 3

  Q <- matrix(c(
    1, 1, 0,
    0, 1, 0,
    1, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    1, 0, 1,
    0, 1, 0,
    0, 0, 1,
    1, 0, 0,
    1, 0, 1,
    1, 0, 1,
    1, 0, 0,
    1, 0, 0,
    0, 0, 1,
    1, 0, 1,
    0, 1, 1,
    0, 0, 1,
    0, 0, 1,
    1, 0, 1,
    1, 0, 1,
    0, 0, 1,
    0, 1, 0,
    0, 1, 0,
    1, 0, 0,
    0, 0, 1,
    1, 0, 0,
    0, 0, 1), nrow=num_items, ncol=num_skills, byrow=TRUE)
  colnames(Q) <- paste("Dim",c(1:num_skills),sep="")
  rownames(Q) <- paste("Item",c(1:num_items),sep="")
  Q = t(Q)

  skill_cov <- matrix(c(.36, .12, .06, .12, .29, -.13, .06, -.13, .26), nrow=3,ncol=3)
  # skill_cov <- diag(3)
  skill_mean <- rep(0,3)
  models <- build_vae_normal_full_covariance(num_items, num_skills,
                                             Q, skill_mean, skill_cov,
                                             enc_hid_arch = c(10,10), kl_weight = 1)
  vae <- models[[3]]
  test_fit_model(models, Q, 10)
}

#' TODO documentation
#' Train the VAE to simulated data and plot some results
#'
#' @param models a list of three models: the encoder, decoder, and vae
#' @param Q a binary matrix relating skills and items
#' @param num_reps the number of data sets to train on
test_fit_model <- function(models, Q, num_reps){
  num_items <- 28
  num_skills <- 3
  num_train <- 9000L
  num_test <- 1000L
  num_epochs <- 10L
  batch_size <- 64L
  a_results <- matrix(0, 37, num_reps)
  b_results <- matrix(0, 28, num_reps)
  data_dir <- 'C:\\Users\\gconverse\\Desktop\\vae_tf_prob_play\\data\\'
  print("Begin training")

  for(r in 1:num_reps){
    Y <- as.matrix(read.csv(file=paste(data_dir,"Yrep",r,".csv",sep=""), sep=";", header=FALSE))		#item response values
    Y <- array(data=Y, dim=c(10000, num_items))
    data_train <- Y[1:num_train,]
    data_test <- Y[num_train:10000,]
    encoder <- models[[1]]
    decoder <- models[[2]]
    vae <- models[[3]]

    #########  Model training
    keras::fit(vae,
      data_train, data_train,
      shuffle = FALSE,
      epochs = num_epochs,
      verbose = 1,
      batch_size = batch_size,
    )

    # Get skill predictions
    skill_preds <- predict(encoder,data_test)
    skill_preds <- skill_preds[[1]]
    # Estimated weights
    W <- keras::get_weights(vae)
    discr <- as.matrix(W[[length(W) - 1]])
    diff <- as.matrix(W[[length(W)]])

    #formatting
    a <- rep(0,37)
    count = 0
    for (row in 1:3){
      for (col in 1:28){
        if (Q[row,col] > 0){
          count = count+1
          a[count] = discr[row,col]
        }
      }
    }
    a_results[,r] = a
    b_results[,r] = diff
  }
  avg_a <- rowMeans(a_results)
  avg_b <- rowMeans(b_results)
  original_a <- as.numeric(read.csv(file=paste(data_dir, 'a_values_reordered.csv', sep = ''), header=FALSE, sep = ';')[,1])
  original_b <- as.numeric(read.csv(file=paste(data_dir, 'b_values.csv', sep = ''), header=FALSE, sep = ';')[,1])
  #re-order to fit data correctly
  avg_a <- avg_a[c(1,4,5,6,7,8,9,10,11,12,13,
  2,3,14,16,17,18,19,15,
  26,27,28,29,30,31,32,33,34,35,36,37,
  20,21,22,23,24,25)]
  # avg_b <- avg_b[c(1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,3,4,5,6,7,8,9)]
  # original_a <- original_a[c(1,12,13,2,3,4,5,6,7,8,9,10,11,
  #                            14,19,15,16,17,18,
  #                            32,33,34,35,36,37,20,21,22,23,24,25,26,27,28,29,30,31)]
  print(cor(avg_a, original_a))
  print(cor(avg_b, original_b))
  plot(original_a, avg_a)
  readline(prompt="Press [enter] for next plot")
  plot(original_b, avg_b)
  readline(prompt="Press [enter] for next plot")
  hist(skill_preds[,1])
  readline(prompt="Press [enter] for next plot")
  hist(skill_preds[,2])
  readline(prompt="Press [enter] for next plot")
  hist(skill_preds[,3])
  readline(prompt="Press [enter] for next plot")
  plot(skill_preds[,1],skill_preds[,2])
  readline(prompt="Press [enter] for next plot")
  plot(skill_preds[,1],skill_preds[,3])
  readline(prompt="Press [enter] for next plot")
  plot(skill_preds[,2],skill_preds[,3])
  print(var(skill_preds[,1]))
  print(var(skill_preds[,2]))
  print(var(skill_preds[,3]))
  print(cov(skill_preds[,1],skill_preds[,2]))
  print(cov(skill_preds[,1],skill_preds[,3]))
  print(cov(skill_preds[,2],skill_preds[,3]))

}
