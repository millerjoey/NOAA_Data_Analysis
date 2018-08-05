import numpy.random as random
import numpy as np

### Directly sample m
def sample_m(mean = 0, variance = 1):
    return(random.randn()*np.sqrt(variance) + mean)
    
# Integrating out \mu_j leads to the returned normal parameters.
def compute_normal_params(s2, X_bar, X_var, hyper_mean, hyper_var): # X_bar, X_var are numpy arrays of the data.
    variance = 1/((1/(X_var + s2)).sum() + 1/hyper_var**2) # Inverse-variances add.
    mean = variance*((X_bar/(X_var + s2)).sum() + hyper_mean/(hyper_var**2))
    return(mean, variance)

#### Metropolis Sampler for s^2

# Some unnormalized log-densities
def ldens_normal(x, mu, var):
    return(-(x - mu)**2/(2*var) - np.log(var)/2)

def ldens_ps(s2, alpha, beta):
    return((-alpha - 1)*np.log(s2) + -beta/s2)

def ldens_px(x_bar, m, x_var, s2):
    return(ldens_normal(x_bar, m, x_var + s2))

def sample_s2(s2, alpha, beta, m, X_bar, X_var, propose_sigma):
    s2_proposed = random.randn()*propose_sigma + s2
    if s2_proposed <= 0:
        return(s2)
    # Test for numeric problems and test value of s^2
    llik_proposal = ldens_px(X_bar, m, X_var, s2_proposed)
    if any(llik_proposal == 0.0):
        return(s2)
    else:
        llik_current = ldens_px(X_bar, m, X_var, s2)
        if np.random.rand() < np.exp(llik_proposal.sum() + 
                                 ldens_ps(s2_proposed, alpha, beta) - 
                                 llik_current.sum() -
                                 ldens_ps(s2, alpha, beta)):
            return(s2_proposed)
        else:
            return(s2)

# Wrapper function

# X_bar ithe array of means
# X_var is the array of variances
def gen_samples(X_bar, X_var, m_init, s2_init, iters = 10000, burn_in = 1000, propose_sigma = 1, hyper_alpha = 1.1, hyper_beta = 1, hyper_mean = 20, hyper_var = 10):
    m_samples = np.zeros(iters + burn_in)
    s2_samples = np.zeros(iters + burn_in)
    m_samples[0] = m_init
    s2_samples[0] = s2_init
    for i in range(1, iters + burn_in):
        # Sample m via direct sampling from a Normal distribution
        mean, variance = compute_normal_params(s2_samples[i-1], X_bar, X_var, hyper_mean, hyper_var)
        m_samples[i] = sample_m(mean, variance)
        # Sample s via Metropolis-Hastings
        s2_samples[i] = sample_s2(s2_samples[i-1], hyper_alpha, hyper_beta, m_samples[i], X_bar, X_var, propose_sigma)
    return(m_samples[burn_in:], s2_samples[burn_in:])
