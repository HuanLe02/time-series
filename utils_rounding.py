import numpy as np
import utils_threshold

def rounding_thresholds(datalist, round_digits, mode='nearest'):
    # check mode
    assert mode in ['nearest', 'floor']
    
    # discretize the state space
    min_val, max_val =  np.min(datalist), np.max(datalist)
    thresholds = None
    
    inc = 10**(-round_digits)
    
    # mode: round to the neareast
    if mode == 'nearest':
        min_round, max_round = np.round(min_val, round_digits), np.round(max_val, round_digits)
        # generate intervals
        thresholds = np.arange(min_round-5*10**(-round_digits-1), max_round+5*10**(-round_digits-1) + inc, inc)
        
    # mode: floor
    elif mode == 'floor':
        min_fl, max_fl = np.floor(min_val*(10**round_digits)), np.floor(max_val*(10**round_digits))
        min_fl = min_fl / (10**round_digits)
        max_fl = max_fl / (10**round_digits)
        thresholds = np.arange(min_fl, max_fl + inc, inc)
    
    # return threshold 1d array
    return thresholds
    