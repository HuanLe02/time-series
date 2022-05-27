import numpy as np
from utils_markov import markov_from_states

# function to discretize the state space
def discretize_thresholds(dataseries, boundaries):
    """
    Input:
        dataseries: time series. NumPy Array of dimensions (n, )
        boundaries: boundary values between states. rank-1 Numpy array
                    categories will be (-inf, boundaries[0]), [boundaries[0], boundaries[1]), ...,
                                       [boundaries[-2], boundaries[-1]), [boundaries[-1], inf)
    Output:
        states: discretized states. NumPy Array of dimensions (n, )
    """
    # copy time series data
    discretized = np.empty(len(dataseries))
    new_bounds = np.concatenate([np.array([-np.inf]),  np.sort(boundaries), np.array([np.inf])]).astype('float64')
    states_dict = {}
    # print(new_bounds)
    
    # transform data into range categories
    state = 0
    for low, high in zip(new_bounds[:-1], new_bounds[1:]):
        idxs = np.where((low <= dataseries) & (dataseries < high))    # indices of items that fall in range
        discretized[idxs] = state
        states_dict[str(state)] = np.array([low, high])
        state += 1
    
    return discretized.astype("int64"), states_dict

def markov_from_data(dataseries, thresholds, noise_mean=0.0, noise_sd=0.0, seed=None, order=1):
    """
    Input:
        dataseries: time series. NumPy Array of dimensions (n, )
        noise_mean, noise_sd: mean and SD of normal distribution of noise
        thresholds: values that separate categories
    Output:
        markov: right Markov matrix
        freq: frequency matrix
    """
    # account for N(noise_mean, noise_sd) noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=noise_mean, scale=noise_sd, size=len(dataseries))
    # discretize datalist
    states_sequence, states_dict = discretize_thresholds(dataseries - noise, thresholds)
    markov = markov_from_states(states_sequence, states_dict, order)
    return markov

def avg_markov_from_data(num_passes, dataseries, thresholds, noise_mean=0.0, noise_sd=0.0, seed=None, order=1):
    assert num_passes >= 1
    
    # get observed states
    observed_states, states_dict = discretize_thresholds(dataseries, thresholds)
    
    # list of bounds(1d np.array)
    bounds_list = np.array(list(states_dict.values()))
    
    # first pass
    sum_markov = markov_from_data(dataseries, thresholds, noise_mean, noise_sd, seed, order)
    # early return
    if (num_passes == 1) or (noise_sd == 0.0):
        return sum_markov, observed_states, states_dict, bounds_list
    
    # loop to calculate average markov matrix
    for i in range(num_passes-1):
        new_markov = markov_from_data(dataseries, thresholds, noise_mean, noise_sd, seed, order)
        sum_markov += new_markov
        
    return sum_markov / num_passes, observed_states, states_dict, bounds_list
