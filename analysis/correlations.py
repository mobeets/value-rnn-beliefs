import numpy as np
from sklearn import preprocessing
import scipy.linalg

#%%  linear regression fitting and evaluation

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
    return {'rsq': rsq}

#%% BELIEF R-SQUARED

def fit_belief_weights(trials):
	pass

def add_belief_prediction(trials, belief_weights):
	pass

def score_belief_rsq(trials):
	pass
