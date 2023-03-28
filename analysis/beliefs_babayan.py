import numpy as np
import scipy
from .beliefs_starkweather import initial_belief, transition_distribution, observation_distribution

def pomdp(reward_times, p_omission=0.0, ITIhazard=1/65., nITI_microstates=1, jitter=1):
    """
    returns transition and observation distributions for cue
    """
    reward_times = np.array(reward_times); reward_times -= 1; # -1 to match starkweather expectations
    ISIpdf = np.ones(len(reward_times))/len(reward_times)
    ISIcdf = np.cumsum(ISIpdf)

    K = reward_times.max() + 1 + nITI_microstates
    ISIhazard = ISIpdf
    ISIhazard[1:] = ISIpdf[1:]/(1-ISIcdf[:-1])
    reward_hazards = ISIhazard
    iti_times = np.arange(reward_times.max()+1, K)
    
    T = transition_distribution(K, reward_times, reward_hazards, p_omission, ITIhazard, iti_times=iti_times)
    O = observation_distribution(K, reward_times, p_omission, ITIhazard, iti_times=iti_times)
    return T, O

def get_states_and_observations(trials, iti_min=0, reward_amounts=None):
    """ n.b. assumes there are only two blocks """
    NULL = 0
    STIM = 1
    REW = 2

    xs = []
    ss = []
    mults = [] # for offsetting everything for each block index
    assert len(reward_amounts) <= 2
    for trial in trials:
        if len(reward_amounts) == 1 or (int(trial.y.max()) == reward_amounts[0]):
            mult = 0
        else:
            mult = 1
        for t in np.arange(trial.trial_length):            
            if t == trial.iti:
                x = STIM
            elif t == trial.iti + trial.isi:
                x = REW
            else:
                x = NULL
            
            if t < trial.iti or t >= (trial.iti + trial.isi):
                s = np.min([t + 1 - iti_min, 0]) # +1 for an off-by-one fix
            else:
                s = t - trial.iti + 1
            if x == REW:
                s = -iti_min # REW means we are in FIRST iti

            xs.append(x)
            ss.append(s)
            mults.append(mult)
    S = np.array(ss)
    s_range = S.max() - S.min()
    S[S <= 0] += (1 + s_range) # make ITI last
    S -= 1 # shift so that lowest state is 0
    
    # now shift everything up for the second block
    M = np.array(mults)
    S[M == 1] += S.max() + 1

    return S, np.array(xs)

def get_beliefs(observations, T, O, prior=0.5, reward_amounts=(1,10), reward_sigma=1,
                ntrials_per_block=None, prior_by_prev_block=None):
    """
    the strategy here is to update beliefs as if there's one block (B)
        while also keeping track of the probability of being in each block (P)
        then duplicate the beliefs, weighting each by P and (1-P), respectively
    """
    if ntrials_per_block is None:
        pass#raise Exception("You need to pass ntrials_per_block to reset beliefs to prior")
    else:
        assert len(np.unique(ntrials_per_block)) == 1
        ntrials_per_block = ntrials_per_block[0]
    if prior_by_prev_block is not None:
        assert len(prior_by_prev_block) == len(reward_amounts)
    
    b = initial_belief(T.shape[0])
    p = prior # we will only keep track of one block
    B = []
    P = []
    eps = 1e-5
    rewards_seen = 0
    last_reward_index_seen = 0
    for i,x in enumerate(observations):
        if x.sum() == 0: # NULL
            o = 0
        elif x[0] > 0: # STIM
            o = 1
        else: # REW
            o = 2
            r = x[-1]
            logp = np.log([np.clip(p,eps,1-eps), 1-np.clip(p,eps,1-eps)])
            logpl = scipy.stats.norm(reward_amounts, reward_sigma).logpdf(r)
            logp += logpl
            p = np.exp(logp - scipy.special.logsumexp(logp))
            p = p[0]
            rewards_seen += 1
            if ntrials_per_block is not None:
                last_reward_index_seen = reward_amounts.index(r)
        if ntrials_per_block is not None and rewards_seen == ntrials_per_block:
            rewards_seen = 0
            if prior_by_prev_block is not None:
                p = prior_by_prev_block[last_reward_index_seen]
            else:
                p = prior

        b = b.T @ (T * O[:,:,o])
        b = b/b.sum()

        B.append(b)
        P.append(p)
    B = np.vstack(B)
    P = np.vstack(P)
    if np.isnan(B).any():
        print("NaN in beliefs. Something went wrong!")    
    B = np.hstack([P*B, (1-P)*B])
    return B, P
