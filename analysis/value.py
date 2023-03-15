import numpy as np

#%% get value weights

def value_weights_tdls(responses, gamma, lambda0=0, add_bias=True, X=None):
    if X is None:
        X = np.vstack([trial.Z for trial in responses])
    if add_bias:
        X = np.hstack([X, np.ones((X.shape[0],1))])
    r = np.vstack([trial.y for trial in responses])[1:]
    X_cur = X[:-1]
    X_next = X[1:]
    
    B_cur = X_cur
    B_next = X_cur - gamma*X_next
    
    X = B_cur.T @ B_next + lambda0*np.eye(B_cur.shape[1])
    y = B_cur.T @ r
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w

#%% add value and rpe

def make_predictions(rs, Z, w, gamma):
    rpes = []
    values = []
    for t in range(1,len(rs)):
        zprev = Z[t-1,:]
        z = Z[t,:]
        rpe = rs[t] + w @ (gamma*z - zprev)
        value = w @ zprev
        
        rpes.append(rpe)
        values.append(value)
    rpes.append(np.nan)
    values.append(np.nan)
    return np.array(rpes), np.array(values)

def add_value_and_rpe(trials, value_weights, gamma):
    Z = np.vstack([trial.Z for trial in trials])
    rs = np.hstack([trial.y for trial in trials])
    rpes, values = make_predictions(rs, Z, value_weights, gamma)
    i = 0
    for trial in trials:
        trial.value = values[i:(i+trial.trial_length)]
        trial.rpe = rpes[i:(i+trial.trial_length-1)]
        i += trial.trial_length
    return trials

#%% analyze

def score_mse(trials, pomdp_trials):
    # score value
    ys = np.hstack([trial.value for trial in trials])
    yhats = np.hstack([trial.value for trial in pomdp_trials])
    value_mse = np.mean((ys - yhats)**2)

    # score rpes
    ys = np.hstack([trial.rpe for trial in trials])
    yhats = np.hstack([trial.rpe for trial in pomdp_trials])
    rpe_mse = np.mean((ys - yhats)**2)
    return {'value_mse': value_mse, 'rpe_mse': rpe_mse}

def analyze(Trials, gamma, pomdp=None):
    weights = value_weights_tdls(Trials['train'], gamma)
    for _, trials in Trials.items():
        add_value_and_rpe(trials, weights, gamma)
    mse = score_mse(Trials['test'], pomdp['Trials']['test'])
    return {'weights': weights, 'mse': mse}
