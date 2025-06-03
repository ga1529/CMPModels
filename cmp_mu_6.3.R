library(cmdstanr)
library(COMPoissonReg)

ecmp_mu <- write_stan_file('functions {
  
  real approximation(real log_mu, real nu){ // Gaunt Approximation
    
    
    real nu_mu = nu * exp(log_mu);
    real nu2 = nu^2;
    // first 4 terms of the residual series
    real log_sum_resid = log1p(
      nu_mu^(-1) * (nu2 - 1) / 24 +
        nu_mu^(-2) * (nu2 - 1) / 1152 * (nu2 + 23) +
        nu_mu^(-3) * (nu2 - 1) / 414720 * (5 * nu2^2 - 298 * nu2 + 11237)+
        nu_mu^(-4) * (nu2 - 1) / 39813120 * (5 * nu2^3 - 1887*nu2^2 - 241041*nu^2 + 2482411)
      
    );
    return nu_mu + log_sum_resid  -
      ((log(2 * pi()) + log_mu) * (nu - 1) / 2 + log(nu) / 2);  
    
  }
  
  // Calculates log S(mu, nu)
  real summation(real log_mu, real nu) {
    real z = negative_infinity();
    real z_last = 0;
    
    for (j in 0:20000) {
      z_last = z;
      z = log_sum_exp(z, nu * (j * log_mu - lgamma(j + 1)));
      
      // Check for convergence, 1e-6
      if (abs(z - z_last) < 1e-6) {
        return z;
      }
    }
    reject("did not converge");
  }
  
  real partial_sum(array[] int slice_n, int start, int end,
                   array[] int y_slice,
                   array[] int jj_slice, array[] int ii_slice, vector theta,
                   vector beta, vector nu, array[] int kk_slice, vector gamma, vector alpha) {
    real partial_target = 0.0;
    
    for (n in start:end) {
      // Linear predictor for log(mu)
      real log_mu = alpha[jj_slice[n]]*theta[ii_slice[n]] + beta[jj_slice[n]] + gamma[kk_slice[n]];
      
      real log_prob = 0;
      
      if (nu[jj_slice[n]]== 0) {
        reject("nu cannot be 0");
      }
      
      if ( (log_mu > log(1.5)) && (nu[jj_slice[n]] * log_mu > log(1.5)) ) {
        
        log_prob =  nu[jj_slice[n]] * (y_slice[n] * log_mu - lgamma(y_slice[n] + 1)) - approximation(log_mu, nu[jj_slice[n]]);
        
      } else {
        
        log_prob =  nu[jj_slice[n]] * (y_slice[n] * log_mu - lgamma(y_slice[n] + 1)) - summation(log_mu,nu[jj_slice[n]]);
      }
      
      partial_target += log_prob;
    }
    return partial_target;
  }
}
data {
  int<lower=0> I; // Number of people
  int<lower=0> J; // Number of items
  int<lower=1> N; // Number of observations
  int<lower=1> K; // Number of item types
  array[N] int<lower=1, upper=I> ii;
  array[N] int<lower=1, upper=J> jj;
  array[N] int<lower=1, upper=K> kk;
  array[J] int<lower=1,upper=K> item_type_for_beta;
  array[N] int<lower=0> y;
  array[N] int seq_N;
  int<lower=0> grainsize;
}

parameters {
  
  vector[I] theta;
  vector[K] gamma;
  vector[J] beta;
  vector<lower=0>[J] alpha;
  vector<lower=0>[J] nu;
  
  real<lower=0> sigma_nu;
  vector<lower=0>[K] sigma_beta_k;
  real<lower=0> sigma_gamma;
  real<lower=0> sigma_alpha;
  
  real mu_gamma;
  
  
}
model {
  
  sigma_nu ~ cauchy(0,5);
  nu ~ lognormal(0, sigma_nu);
  
  theta ~ normal(0, .3); 
  
  
  alpha ~ lognormal(0,sigma_alpha);
  sigma_alpha ~ cauchy(0,5);
  
  gamma ~ normal(mu_gamma,sigma_gamma);
  mu_gamma ~ normal(0,5);
  sigma_gamma ~ cauchy(0,5);
  
  sigma_beta_k ~ cauchy(0,5);
  
  for (j in 1:J) { 
    beta[j] ~  normal(gamma[item_type_for_beta[j]], sigma_beta_k[item_type_for_beta[j]]);
  }
  
  target += reduce_sum(partial_sum, seq_N, grainsize, y, jj, ii, theta,
                       beta, nu, kk, gamma, alpha);
}

generated quantities {
  array[N] real log_lik;
  
  for (n in 1:N) {
    real log_mu = alpha[jj[n]]*theta[ii[n]] + beta[jj[n]] + gamma[kk[n]];
    
    if ( log_mu > log(1.5)  &&   nu[jj[n]] * log_mu > log(1.5) ) {
      
      log_lik[n] =  nu[jj[n]] * (y[n] * log_mu - lgamma(y[n] + 1)) - approximation(log_mu, nu[jj[n]]);
    } else {
      
      log_lik[n] =  nu[jj[n]] * (y[n] * log_mu - lgamma(y[n] + 1)) - summation(log_mu,nu[jj[n]]);
    }
  }
}

')

mod_ecmp <- cmdstan_model(ecmp_mu, compile = T, cpp_options = list(stan_threads=T))


I <- 100
J <- 24
K <- 2

N <- I * J
ii <- rep(1:I, times = J)
jj <- rep(1:J, each = I)
kk <- rep(1:K, times = N / K)
item_type_for_beta <- kk[1:J]
beta <- numeric(length = J)

theta <- rnorm(I, 0, .3)
gamma <- rnorm(K, 2, .5)
alpha <- rlnorm(J, 0, .25)
nu <- rlnorm(J,0, .25)
#unique_sd <- runif(length(unique(item_type_for_beta)), 0, 1)
#unique_sd <- c(0, .5)
#gamma[2] <- 1

z <- 0
for (type in unique(item_type_for_beta)) {
  # Get the index of items for this type
  indices <- which(item_type_for_beta == type)
  # Get the corresponding gamma value for this type
  gamma_value <- gamma[type]
  
  sd_value <- unique_sd[type]
  
  beta[indices] <- rnorm(length(indices), mean = gamma_value, sd = sd_value)
  z <- z + 1
  
}
set.seed(Sys.time())
set.seed(37)
y <- numeric(N)
for(n in 1:N) {
  mu_n <- exp(nu[jj[n]] * (alpha[jj[n]] * theta[ii[n]] + beta[jj[n]] + gamma[kk[n]]))
  y[n] <- rcmp(1, mu_n, nu[jj[n]])
}

stan_data <- list(I = I,
                  J = J,
                  N = N,
                  K = K,
                  kk = kk,
                  item_type_for_beta = item_type_for_beta,
                  ii = ii,
                  jj = jj,
                  y = y,
                  grainsize = 1,
                  seq_N = 1:N)

fit_ecmp_mu <- mod_ecmp$sample(data = stan_data, chains = 1, threads_per_chain = 15,
                               iter_warmup = 700, iter_sampling = 700 ,refresh = 10, thin = 1,
                               max_treedepth = 10, save_warmup = T, adapt_delta = .85, seed = 38 ) 

fit_ecmp_mu$summary("sigma_beta_k")
unique_sd


hist(y)
beta[which(item_type_for_beta== 3)]


gamma


# as unique sd increases, overestimation increases (est > real)
# large unique sd with smaller gamma (as in gamma = unique_sd) gives really slow sampling
