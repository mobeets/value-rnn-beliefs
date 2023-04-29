import numpy as np
from sklearn import preprocessing
import scipy.linalg

#%% RDMs and RSA

def get_rsa(trials, rdm_method='euclidean', rsa_method='cosine'):
    import rsatoolbox
    Z = np.vstack([trial.Z for trial in trials])
    B = np.vstack([trial.B for trial in trials])
    S = np.hstack([trial.S for trial in trials])
    all_states = np.unique(S)
    Zavg = np.vstack([Z[S == s].mean(axis=0) for s in all_states])
    Bavg = np.vstack([B[S == s].mean(axis=0) for s in all_states])
    data_z = rsatoolbox.data.Dataset(Zavg,
        obs_descriptors={'stimulus': all_states})
    rdm_z = rsatoolbox.rdm.calc_rdm(data_z,
        method=rdm_method, descriptor=None, noise=None)
    data_b = rsatoolbox.data.Dataset(Bavg,
        obs_descriptors={'stimulus': all_states})
    rdm_b = rsatoolbox.rdm.calc_rdm(data_b,
        method=rdm_method, descriptor=None, noise=None)
    rsa = rsatoolbox.rdm.compare(rdm_z, rdm_b, method=rsa_method)[0][0]
    return rdm_z, rsa

#%% linear regression fitting and evaluation

def linreg_fit(X, Y, scale=False, add_bias=True):
    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    else:
        scaler = None
    if add_bias:
        Z = np.hstack([X, np.ones((X.shape[0],1))])
    else:
        Z = X
    W = scipy.linalg.lstsq(Z, Y)[0]
    Yhat = Z @ W
    return {'W': W, 'scaler': scaler, 'scale': scale, 'add_bias': add_bias}

def linreg_eval(X, Y, mdl):
    if mdl['scaler']:
        X = mdl['scaler'].transform(X)
    if mdl['add_bias']:
        Z = np.hstack([X, np.ones((X.shape[0],1))])
    else:
        Z = X
    Yhat = Z @ mdl['W']
    
    # get r-squared
    top = Yhat - Y
    bot = Y - Y.mean(axis=0)[None,:]
    rsq = 1 - np.diag(top.T @ top).sum()/np.diag(bot.T @ bot).sum()
    return {'Yhat': Yhat, 'rsq': rsq}

#%% BELIEF R-SQUARED

def fit_belief_weights(trials):
    X = np.vstack([trial.Z for trial in trials])
    Y = np.vstack([trial.B for trial in trials])
    return linreg_fit(X, Y, scale=True, add_bias=True)

def add_and_score_belief_prediction(trials, belief_weights):
    X = np.vstack([trial.Z for trial in trials])
    Y = np.vstack([trial.B for trial in trials])
    res = linreg_eval(X, Y, belief_weights)

    # add belief prediction to trials
    Yhat = res['Yhat']
    i = 0
    for trial in trials:
        trial.Bhat = Yhat[i:(i+trial.trial_length)]
        i += trial.trial_length
    return res['rsq']

def analyze(model, Trials):
    results = {}
    results['weights'] = fit_belief_weights(Trials['train'])
    results['rsq'] = add_and_score_belief_prediction(Trials['test'], results['weights'])
    results['rdm'], results['rsa'] = get_rsa(Trials['test'])
    return results
