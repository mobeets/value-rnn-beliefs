import numpy as np

#%% find value weights using TD-LS

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
    return w[:,0]

#%% add value and rpe to trials

def make_predictions(rs, Z, w, gamma, add_bias=True):
    if add_bias:
        Z = np.hstack([Z, np.ones((Z.shape[0],1))])
    
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
    rs = np.hstack([trial.y[:,0] for trial in trials])
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
    ys = np.hstack([trial.value for trial in trials])[:-1] # last is nan
    yhats = np.hstack([trial.value for trial in pomdp_trials])[:-1]
    value_mse = np.mean((ys - yhats)**2)

    # score rpes
    ys = np.hstack([trial.rpe for trial in trials])
    yhats = np.hstack([trial.rpe for trial in pomdp_trials])
    rpe_mse = np.mean((ys - yhats)**2)
    return {'value_mse': value_mse, 'rpe_mse': rpe_mse}

def rpe_summary(E, trials):
    if 'starkweather' in E.experiment_name:
        rpes_cue = [trial.rpe[(trial.iti - E.iti_min):(trial.iti + trial.isi - 1)] for trial in trials if trial.y.max() > 0]
        n = max([len(x) for x in rpes_cue])
        rpes_cue = np.nanmean(np.vstack([np.hstack([rpe, np.nan*np.ones(n-len(rpe))]) for rpe in rpes_cue]), axis=0)
        rpes_end = [trial.rpe[:(trial.iti - E.iti_min)] for trial in trials if trial.y.max() > 0]
        n = max([len(x) for x in rpes_end])
        rpes_end = np.nanmean(np.vstack([np.hstack([rpe, np.nan*np.ones(n-len(rpe))]) for rpe in rpes_end]), axis=0)
        rpes = [(trial.isi if trial.y.max() > 0 else np.nan, trial.rpe[-1]) for trial in trials]
        rpes_mu = {}
        isis = sorted(np.unique([isi for isi,rpe in rpes if ~np.isnan(isi)]))
        for isi in isis:
            rpes_mu[isi] = []
        for isi,rpe in rpes:
            if np.isnan(isi):
                continue
            if isi not in rpes_mu:
                rpes_mu[isi] = []
            rpes_mu[isi].append(rpe)
        for isi in rpes_mu:
            rpes_mu[isi] = np.mean(rpes_mu[isi])
        return {'rpes_cue': rpes_cue, 'rpes_end': rpes_end, 'rpes_mu': rpes_mu}
    elif E.experiment_name == 'babayan':
        rpes = {}
        for c, alignType in enumerate(['CS', 'US']):
            crpes = {}
            if alignType == 'CS':
                getter = lambda trial: trial.rpe[trial.iti-1]
            else:
                getter = lambda trial: trial.rpe[trial.iti+trial.isi-1]
            for group in [1,2,3,4]:
                if group == 1:
                    matcher = lambda trial: trial.block_index == 0 and trial.prev_block_index == 0
                elif group == 2:
                    matcher = lambda trial: trial.block_index == 0 and trial.prev_block_index == 1
                elif group == 3:
                    matcher = lambda trial: trial.block_index == 1 and trial.prev_block_index == 0
                else:
                    matcher = lambda trial: trial.block_index == 1 and trial.prev_block_index == 1

                cdata = [(trial.rel_trial_index, getter(trial)) for trial in trials if matcher(trial)]
                mus = []
                ses = []
                ts = range(1,max(E.ntrials_per_block)+1)
                for t in ts:
                    ys = [y for x,y in cdata if x==(t-1)]
                    mu = np.mean(ys)
                    se = np.std(ys)/np.sqrt(len(ys))
                    mus.append(mu)
                    ses.append(se)
                crpes[group] = {'times': ts, 'mus': mus, 'ses': ses}
            rpes[alignType] = crpes
        return rpes
    elif E.experiment_name == 'babayan-interpolate':
        rpes = {}
        for c,ti in enumerate([1,2]):
            for r in E.reward_sizes_per_block:
                crpes = [trial.rpe[-1] for trial in trials if trial.reward_size == r and hasattr(trial, 'rel_trial_index') and trial.rel_trial_index==(ti-1)]
                rpes[(ti,r)] = np.mean(crpes)
        return rpes

def analyze(experiments, Trials, gamma, pomdp=None):
    weights = value_weights_tdls(Trials['train'], gamma)
    for _, trials in Trials.items():
        add_value_and_rpe(trials, weights, gamma)
    results = {'weights': weights}
    
    # summarize rpes at odor and reward
    results['rpe_summary'] = rpe_summary(experiments['test'], Trials['test'])

    # score rpes and value relative to pomdp
    if pomdp is not None:
        results['mse'] = score_mse(Trials['test'], pomdp['Trials']['test'])
    return results
