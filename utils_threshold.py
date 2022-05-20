import numpy as np
from utils_markov import markov_from_states

# function to discretize the state space
def discretize_thresholds(dataseries, boundaries):
    """
    Input:
        dataseries: time series. NumPy Array of dimensions (n, )
        boundaries: boundary values between states. List or rank-1 Numpy array
                    categories will be (-inf, boundaries[0]], (boundaries[0], boundaries[1]], ...,
                                       (boundaries[-2], boundaries[-1]], (boundaries[-1], inf)
    Output:
        states: discretized states. NumPy Array of dimensions (n, )
    """
    # copy time series data
    discretized = np.copy(dataseries)
    new_bounds = np.concatenate([np.array([-np.inf]), boundaries, np.array([np.inf])])
    
    # transform data into range categories
    state = 0
    for low, high in zip(new_bounds[:-1], new_bounds[1:]):
        discretized[(low < discretized) & (discretized <= high)] = state
        state += 1
    
    return discretized.astype("int64"), new_bounds

def markov_from_data(dataseries, thresholds, noise_mean=0.0, noise_variance=1.0, seed=None, order=1):
    """
    Input:
        dataseries: time series. NumPy Array of dimensions (n, )
        noise_mean, noise_variance: mean and variance of normal distribution of noise
        thresholds: values that separate categories
    Output:
        markov: right Markov matrix
        freq: frequency matrix
    """
    # account for N(noise_mean, noise_variance) noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=noise_mean, scale=noise_variance, size=len(dataseries))
    # discretize datalist
    observed_states, _ = discretize_thresholds(dataseries + noise,thresholds)
    return markov_from_states(observed_states, order=order)

def avg_markov_from_data(num_passes, dataseries, thresholds, noise_mean=0.0, noise_variance=1.0, seed=None, order=1):
    sum_markov = 0.
    for i in range(num_passes):
        new_markov, _ = markov_from_data(dataseries, thresholds, noise_mean, noise_variance, seed, order)
        sum_markov += new_markov
    return sum_markov / num_passes