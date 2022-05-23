import numpy as np

# generate markov from observed states
def markov_from_states(observed_states, states_dict, order=1):   
    """
    Input:
        observed_states: list of discretized states
        order: how far to go back (i.e. state t depends on state t-order)
    Output:
        markov: right Markov matrix
        freq: frequency matrix
    """
    # count the transitions
    # data_int = observed_states - np.min(observed_states)
    n_states = len(states_dict.keys())
    pre = observed_states[:-order]
    post = observed_states[order:]
    markov = np.zeros((n_states, n_states))
    for a,b in zip(pre,post):
        markov[a, b] += 1.
    
    # save a frequency matrix copy
    # frequencies = np.copy(markov).astype('int64')
    
    # account for all-zero rows
    for i in np.where(np.sum(markov, 1) == 0.)[0]:
        markov[i][i] = 1.
    
    # normalize to keep sum of rows = 1
    markov = markov / np.sum(markov, axis=1, keepdims=True)    
    
    return markov #, frequencies

def random_walk_weighted(n_steps, markov, observed_states):        # random walk but weighted
    log_prob = 0.
    rng = np.random.default_rng()
    state = observed_states[-1]
    min_state = np.min(observed_states)
    future_states = [state]

    for i in range(n_steps):
        # weightes associated w/ each states
        weights = markov[state - min_state]
        # possible transitions are all states
        transitions = np.arange(len(weights))
        
        # select new state
        new_state = rng.choice(transitions, p=weights) + min_state
        # add to cumulative log-prob
        log_prob += np.log(markov[state-min_state, new_state-min_state])
        future_states.append(new_state)
        state = new_state
    
    return {'log_prob': log_prob, 'path': np.array(future_states)}