import numpy as np

# generate markov from observed states
def markov_from_states(states_sequence, states_dict, order=1):   
    """
    Input:
        states_sequence: sequence of discretized states
        order: how far to go back (i.e. state t depends on state t-order)
    Output:
        markov: right Markov matrix
        freq: frequency matrix
    """
    # count the transitions
    # data_int = states_sequence - np.min(states_sequence)
    n_states = len(states_dict.keys())
    pre = states_sequence[:-order]
    post = states_sequence[order:]
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

def random_walk_weighted(n_steps, markov, states_sequence):        # random walk but weighted
    """
        State must be 0-indexed
    """
    log_prob = 0.
    rng = np.random.default_rng()
    state = states_sequence[-1]    # last state
    future_states = [state]

    for i in range(n_steps):
        # weights associated w/ each states
        weights = markov[state]
        # possible transitions are all states
        transitions = np.arange(len(weights))
        
        # select new state
        new_state = rng.choice(transitions, p=weights)
        # add to cumulative log-prob
        log_prob += np.log(markov[state, new_state])
        future_states.append(new_state)
        state = new_state
    
    return {'log_prob': log_prob, 'path': np.array(future_states)}