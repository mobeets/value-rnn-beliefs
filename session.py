import copy
import numpy as np
from valuernn.train import make_dataloader, probe_model
import analysis.value
import analysis.correlations
import analysis.decoding
import analysis.dynamics

def get_activity(model, experiment, sigma=0):
    if model['model_type'] == 'pomdp':
        for trial in experiment.trials:
            trial.Z = trial.B
        trials = experiment.trials
    else:
        dataloader = make_dataloader(experiment, batch_size=1)
        trials = probe_model(model['model'], dataloader)
    trials = trials[1:]

    # add noise (proportional to std. dev of activity across trials)
    if sigma > 0:
        sds = np.vstack([trial.Z for trial in trials]).std(axis=0)
        for trial in trials:
            trial.Z += sigma*sds*np.random.randn(*trial.Z.shape)
    
    return trials

def analyze(model, experiments, pomdp=None, sigma=0, verbose=True):
    if verbose:
        print("Analyzing {}...".format(model['model_type']))
    session = dict((key, val) for key,val in model.items() if key != 'model')
    session['sigma'] = sigma if model['model_type'] != 'pomdp' else 0.0

    # add model activity
    if verbose:
        print("    Adding model activity.")
    Trials = {}
    for name, experiment in experiments.items():
        Trials[name] = get_activity(model, experiment, session['sigma'])
    session['Trials'] = Trials
    
    session['results'] = {}

    # fit value weights and get rpes
    if verbose:
        print("    Analyzing value and rpes.")
    session['results']['value'] = analysis.value.analyze(experiments, Trials, gamma=model['gamma'], pomdp=pomdp)

    # fit belief weights
    if model['model_type'] != 'pomdp':
        if verbose:
            print("    Performing belief regression.")
        session['results']['belief_regression'] = analysis.correlations.analyze(model, Trials)

    # fit decoding weights
    if verbose:
        print("    Performing state decoding.")
    session['results']['state_decoding'] = analysis.decoding.analyze(model, Trials)

    # characterize dynamics
    if model['model_type'] != 'pomdp':
        if verbose:
            print("    Analyzing dynamics.")
        session['results']['memories'] = analysis.dynamics.analyze(model, Trials)

    return session
