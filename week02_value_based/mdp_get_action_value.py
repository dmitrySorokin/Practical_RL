
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    # YOUR CODE HERE
    total = 0
    for s_next in state_values:
        p = mdp.get_transition_prob(state, action, s_next)
        r = mdp.get_reward(state, action, s_next)
        gv = gamma * state_values[s_next]
        
        total += p * (r + gv)

    return total
