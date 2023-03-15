import numpy as np
import scipy.stats

NULL = 0
STIM = 1
REW = 2

def transition_distribution(K, reward_times, reward_hazards, p_omission, ITIhazard, iti_times=None):
    """
    T[i,j] = P(s'=j | s=i)
    """
    if iti_times is None:
        ITI_start = -1
        iti_times = []
    else:
        ITI_start = iti_times[0]
        assert(iti_times[-1] == K-1), 'last iti time should be last state'
    T = np.zeros((K,K))
    
    # no probability of transitioning out of isi during this time
    for k in np.arange(min(reward_times)):
        T[k,k+1] = 1.0
    
    # isi
    for t,h in zip(reward_times, reward_hazards):
        T[t,t+1] = 1-h
        T[t,ITI_start] = h
    # T[-2,-1] = 1
    T[reward_times.max(),ITI_start] = 1
    
    # iti
    for t in iti_times[:-1]:
        T[t,t+1] = 1

    # transitions out of last iti:
    T[-1,-1] = 1 - ITIhazard # 1-(ITIhazard*(1-p_omission)) # = 1- (itih - itih*p_omit) = 1 - itih + itih*p_omit
    T[-1,ITI_start] += ITIhazard*p_omission # += so that when ITI_start == -1, we add probs
    T[-1,0] = ITIhazard*(1-p_omission)
    
    return T

def observation_distribution(K, reward_times, p_omission, ITIhazard, iti_times=None):
    """
    P(x'=m | s=i, s'=j), for m in {NULL, STIM, REW}
    """    
    O = np.zeros((K,K,3))
    if iti_times is None:
        ITI_start = -1
    else:
        ITI_start = iti_times[0]
        assert(iti_times[-1] == K-1), 'last iti time should be last state'
    
    # progressed through time
    for k in np.arange(K-1):
        O[k,k+1,:] = [1,0,0]
    
    # obtained reward
    O[reward_times,ITI_start,:] = [0,0,1]
    
    # stim onset
    O[-1,0,:] = [0,1,0]
    
    # iti
    if np.arange(K)[ITI_start] == K-1:
        O[-1,-1,NULL] = 1-(ITIhazard*p_omission) # stayed in iti
        O[-1,-1,STIM] = ITIhazard*p_omission # omission trial
        O[-1,-1,REW] = 0 # never happens
    else:
        O[-1,-1,:] = [1,0,0] # will always see NULL
        if p_omission > 0:
            O[-1,ITI_start,:] = [0,1,0] # will always see STIM
    return O

def pomdp(cue=0, p_omission=0.1, bin_size=0.2, ITIhazard=1/65., nITI_microstates=1):
    """
    returns transition and observation distributions for cue
    """
    assert cue == 0
    rts = np.arange(1.2, 3.0, 0.2)
    reward_times = (rts/0.2).astype(int)
    ISIpdf = scipy.stats.norm.pdf(rts, rts.mean(), 0.5)
    ISIpdf = ISIpdf/ISIpdf.sum()

    K = reward_times.max() + 1 + nITI_microstates
    ISIcdf = np.cumsum(ISIpdf)
    ISIhazard = ISIpdf
    ISIhazard[1:] = ISIpdf[1:]/(1-ISIcdf[:-1])
    reward_hazards = ISIhazard
    # reward_times = np.round(reward_times / bin_size, 6).astype(int)-1
    iti_times = np.arange(reward_times.max()+1, K)
    
    T = transition_distribution(K, reward_times, reward_hazards, p_omission, ITIhazard, iti_times=iti_times)
    O = observation_distribution(K, reward_times, p_omission, ITIhazard, iti_times=iti_times)
    return T,O

def get_states_and_observations(trials, cue=0, iti_min=0):
    xs = []
    ss = []
    for i,trial in enumerate(trials):
        if trial.cue != cue:
            continue
        lastTrialWasOmission = trials[i-1].y.sum() == 0
        for t in np.arange(trial.trial_length):            
            if t == trial.iti:
                x = STIM
            elif t == trial.iti + trial.isi:
                x = REW
            else:
                x = NULL
            
            if t >= trial.iti and trial.y.sum() == 0: # omission
                s = np.min([t - trial.iti - iti_min, 0])
                if x == REW:
                    x = NULL
            else:
                if t < trial.iti and lastTrialWasOmission:
                    s = 0
                elif t < trial.iti or t >= (trial.iti + trial.isi):
                    s = np.min([t + 1 - iti_min, 0]) # +1 for an off-by-one fix
                else:
                    s = t - trial.iti + 1
                if x == REW:
                    s = -iti_min # REW means we are in FIRST iti
            xs.append(x)
            ss.append(s)
    S = np.array(ss)
    s_range = S.max() - S.min()
    S[S <= 0] += (1 + s_range) # make ITI last
    S -= 1 # shift so that lowest state is 0

    return S, np.array(xs)

def initial_belief(K, iti_min=0):
    b = np.zeros(K)
    b[-(iti_min+1)] = 1.0 # start knowing we are in ITI
    return b

def get_beliefs(observations, T, O):
    b = initial_belief(T.shape[0])
    B = []
    for i,x in enumerate(observations):
        b = b.T @ (T * O[:,:,x])
        b = b/b.sum()
        B.append(b)
    B = np.vstack(B)
    if np.isnan(B).any():
        print("NaN in beliefs. Something went wrong!")
    return B
