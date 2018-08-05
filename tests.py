import numpy as np
import mcmc
from scipy.special import gamma

def test_ldens_normal():   
    p_dx = np.exp(mcmc.ldens_normal(np.arange(-100, 100, 0.0001), 10, 5)).sum()/(np.sqrt(2*np.pi))*0.0001
    assert np.abs(p_dx - 1) < 0.0001

def test_ldens_px():
    p_dx = np.exp(mcmc.ldens_px(np.arange(-100, 100, 0.0001), 10, 3, 2)).sum()/(np.sqrt(2*np.pi))*0.0001
    assert np.abs(p_dx - 1) < 0.0001

def test_compute_normal_params():
    mean, variance = mcmc.compute_normal_params(0, np.array([20, 20]), np.array([1, 2]), hyper_mean = 20, hyper_var = 10)
    assert mean == 20
    assert variance == 1/1.51

def test_ldens_ps():
    p_dx = 5**10/gamma(10)*np.exp(mcmc.ldens_ps(np.arange(0.0001, 100, 0.0001), 10, 5)).sum()*0.0001
    assert np.abs(p_dx - 1) < 0.0001
    
def test_sample_s2(): # Test special case with one obs where X_var = 0: should be Inv-Gamma(alpha + 1/2, beta + (x - mean)/2). Will test for the mean.
    s2_samples = np.zeros(10000)
    s2_samples[0] = 0.6
    for i in range(1,10000):
        s2_samples[i] = mcmc.sample_s2(s2 = s2_samples[i-1], alpha = 10, beta = 5, m = 20, X_bar = np.array([21]), X_var = np.zeros(1), propose_sigma = 0.3)
    post_alpha, post_beta = 10 + 1/2, 5 + (1/2)*(21 - 20)**2
    assert np.abs(np.mean(s2_samples) - post_beta/(post_alpha - 1)) < 0.01

def test_gen_samples():
    # Testing mean of m_samples. Integrated out s^2 and will compare to a grid-approximation for P(m|x).
    
    # Via mcmc code
    x = 21
    alpha = 2
    beta = 1
    
    m_samples, s2_samples = mcmc.gen_samples(X_bar = np.array([x]), X_var = np.zeros(1), m_init = 20, s2_init = 0.9, hyper_alpha = alpha, hyper_beta = beta, burn_in = 10000, iters = 100000, hyper_mean = 20, hyper_var = 10)
    simulated_mean_m = np.mean(m_samples)
    
    # Grid approx
    m_grid = np.arange(-10, 50, 0.0001)
    pm_vals = (gamma(alpha + 1/2)/(beta + 0.5*(x - m_grid)**2)**(alpha + 1/2))*np.exp(-(m_grid - 20)**2/(2*(10**2)))
    grid_mean_m = ((pm_vals/np.sum(pm_vals))*m_grid).sum()
    assert np.abs(simulated_mean_m - grid_mean_m) < 0.01

# If tests fail, run again. The final two are probabilistic.