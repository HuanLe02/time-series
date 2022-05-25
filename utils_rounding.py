import numpy as np
import utils_threshold

def rounding_thresholds(datalist, round_digits):
    # discretize the state space
    round_digits = 1
    min_val, max_val =  np.min(datalist), np.max(datalist)
    min_round, max_round = np.round(min_val, round_digits), np.round(max_val, round_digits)

    # generate intervals
    inc = 10**(-round_digits)
    thresholds = np.arange(min_round-5*10**(-round_digits-1), max_round+5*10**(-round_digits-1) + inc, inc)
    
    return thresholds
    