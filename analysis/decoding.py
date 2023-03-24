#%%

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy.special import softmax
from sklearn.decomposition import PCA

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

def get_state_data(trials):
    trial_keeper = lambda trial: True if not hasattr(trial, 'rel_trial_index') else trial.rel_trial_index > 0
    X = np.vstack([trial.Z for trial in trials if trial_keeper(trial)])
    S = np.hstack([trial.S for trial in trials if trial_keeper(trial)])
    return X, S

def fit_state_weights(trials, model_type, pca=None):
    X, S = get_state_data(trials)
    if pca is not None:
        X = pca.transform(X)
    return decode_X_from_y_fit(X, S)

def score_state_LL(trials, state_weights, model_type, pca=None):
    X, S = get_state_data(trials)
    if model_type == 'pomdp':
        return decode_X_from_y_eval_pomdp(X, S)
    else:
        if pca is not None:
            X = pca.transform(X)
        return decode_X_from_y_eval(X, S, state_weights)

def analyze(model, Trials, usePCs=True):
    if usePCs and model['model_type'] != 'pomdp':
        Z = np.vstack([trial.Z for trial in Trials['train']])
        pca = PCA(n_components=Z.shape[1])
        pca.fit(Z)
    else:
        pca = None
    if model['model_type'] != 'pomdp':
        state_weights = fit_state_weights(Trials['train'], model['model_type'], pca=pca)
    else:
        state_weights = None
    results = score_state_LL(Trials['test'], state_weights, model['model_type'], pca=pca)
    results['state_weights'] = state_weights
    return results
