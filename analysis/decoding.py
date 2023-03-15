#%%

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy.special import softmax

#%%

def to_categorical(Y):
    if len(Y.shape) > 1 and Y.shape[1] > 1:
        return np.argmax(Y, axis=1)
    else:
        return Y.copy()

def decode_X_from_y_fit(X, y, scale=True, class_weight=None, C=1):
    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    else:
        scaler = None
    
    clf = LogisticRegression(multi_class='multinomial',
                max_iter=int(1e4), C=C,
                class_weight=class_weight)
    clf.fit(X, y)
    return {'scaler': scaler, 'clf': clf}

def safelog(x):
    y = x.copy()
    y[y == 0] = np.finfo(np.float32).eps
    return np.log(y)

def decode_X_from_y_eval(X, y, mdl):
    clf = mdl['clf']
    scaler = mdl['scaler']
    if scaler:
        X = scaler.transform(X)
    
    # phat = softmax(clf.decision_function(X))
    
    classes = np.unique(y)
    pte_hat = softmax(clf.decision_function(X), axis=-1)
    if len(pte_hat.shape) == 1:
        pte_hat = np.vstack([pte_hat, 1-pte_hat]).T
    Yh = to_categorical(pte_hat)
    phat_mean = np.vstack([pte_hat[y == c].mean(axis=0) if (y == c).sum() > 0 else np.zeros(len(classes)) for c in classes])
    
    LL = np.hstack([safelog(pte_hat[y == c,c]) for c in np.unique(y)]).mean()
    pcor = 100*np.mean(y == Yh)
    return {'LL': LL, 'pcor': pcor, 'phat_mean': phat_mean}

def decode_X_from_y_eval_pomdp(X, y):
    classes = np.unique(y)
    pte_hat = X.copy()
    assert np.isclose(pte_hat.sum(axis=1), 1).mean()
    pte_hat = pte_hat / pte_hat.sum(axis=1)[:,None]
    Yh = to_categorical(pte_hat)
    phat_mean = np.vstack([pte_hat[y == c].mean(axis=0) if (y == c).sum() > 0 else np.zeros(len(classes)) for c in classes])
    
    LL = np.hstack([safelog(pte_hat[y == c,c]) for c in np.unique(y)]).mean()
    pcor = 100*np.mean(y == Yh)
    return {'LL': LL, 'pcor': pcor, 'phat_mean': phat_mean}

#%% STATE DECODING

def fit_state_weights(trials):
	pass

def score_state_LL(trials, state_weights):
	pass

def analyze(model, Trials, pomdp=None):
    return
