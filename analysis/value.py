import numpy as np

#%%

def value_weights_tdls(responses, gamma, lambda0=0, add_bias=True, X=None):
    if X is None:
        X = np.vstack([trial.Z for trial in responses])
    if add_bias:
        X = np.hstack([X, np.ones((X.shape[0],1))])
    r = np.vstack([trial.y for trial in responses])[1:]
    X_cur = X[:-1]
    X_next = X[1:]
    
    B_cur = X_cur # np.vstack([b0, B[:-1,:]])
    B_next = X_cur - gamma*X_next
    
    X = B_cur.T @ B_next + lambda0*np.eye(B_cur.shape[1])
    y = B_cur.T @ r
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w

#%% VALUE/RPE

def fit_value_weights(trials):
	return

def add_value_and_rpe(trials, value_weights):
	pass

def score_rpe_mse(trials):
	pass
