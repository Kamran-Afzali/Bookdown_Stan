data{
  int N_train;             //  training observations
  int N_test;              // test observations
  int N_pred;              //  predictor variables
  vector[N_train] y_train; // training outcomes
  matrix[N_train, N_pred] X_train; // training data
  matrix[N_test, N_pred] X_test;   // testing data
}
parameters{
  real alpha;           // intercept
  real<lower=0> sigma;   // error SD
  real<lower=0> sigma_B; // hierarchical SD across betas
  vector[N_pred] beta;   // regression beta weights
}
model{
  // group-level (hierarchical) SD across betas
  sigma_B ~ cauchy(0, 1);
  
  // model error SD
  sigma ~ normal(0, 1);
  
  // beta prior 
  beta ~ normal(0, sigma_B);
  
  // model likelihood
  y_train ~ normal(alpha+X_train*beta, sigma);
}
generated quantities{ 
  real y_test[N_test]; // test data predictions
  for(i in 1:N_test){
    y_test[i] = normal_rng(alpha+ X_test[i,] * beta, sigma);
  }
}
