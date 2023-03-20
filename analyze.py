import os.path
import glob
import json
import pickle
import argparse
import numpy as np

from valuernn.tasks import starkweather, babayan
from valuernn.model import ValueRNN
import analysis.beliefs
import session

RECURRENT_CELL = 'GRU'
TRAIN_SEED = 456
TEST_SEED = 123
ITI_P = 1/8
ITI_MIN = 10
GAMMA = 0.93
P_OMISSION = 0.1 # starkweather only
REWARD_TIME = 10 # babayan only
ESN_GAINS = np.arange(0.1, 2.8, 0.2)

def get_experiment(name, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if 'starkweather' in name:
        E = starkweather.Starkweather(ncues=1,
            ntrials_per_cue=1000,
            ntrials_per_episode=1000,
            omission_probability=P_OMISSION if 'task2' in name else 0.0,
            iti_p=ITI_P, iti_min=ITI_MIN, t_padding=0)
    elif name == 'babayan':
        E = babayan.Babayan(nblocks=(100,100), # 1000 trials total
            ntrials_per_block=(5,5),
            reward_sizes_per_block=(1,10),
            reward_times_per_block=(REWARD_TIME,REWARD_TIME),
            jitter=1, # lets reward time vary +/- jitter
            iti_p=ITI_P, iti_min=ITI_MIN,
            include_unique_rewards=False,
            ntrials_per_episode=None) # defaults to len(trials)
    else:
        raise Exception("Unrecognized experiment: {}".format(name))
    E.experiment_name = name
    return E

def get_experiments(name):
    expt_tr = analysis.beliefs.add_states_and_beliefs(name, get_experiment(name, TRAIN_SEED))
    expt_te = analysis.beliefs.add_states_and_beliefs(name, get_experiment(name, TEST_SEED))
    return {'train': expt_tr, 'test': expt_te}

def get_modelfiles(experiment_name, indir, hidden_size=None):
    if 'starkweather' in experiment_name:
        model_name_templates = ['*starkweather*']
        ignore_templates = ['*_initial*', '*babayan*']
        if 'task1' in experiment_name:
            ignore_templates.append('*task2*')
        elif 'task2' in experiment_name:
            ignore_templates.append('*task1*')
    elif experiment_name == 'babayan':
        model_name_templates = ['*babayan*']
        ignore_templates = ['*_initial*', '*starkweather*']
    if hidden_size is not None:
        ignore_templates.append('*h{}*'.format(hidden_size))
    
    for model_name_template in  model_name_templates:
        modelfiles = glob.glob(os.path.join(indir, model_name_template + '.json'))
        for ignore_template in ignore_templates:
            modelfiles = list(set(modelfiles) - set(glob.glob(os.path.join(indir, ignore_template + '.json'))))
    return sorted(modelfiles)

def rnn_model_is_valid(experiment_name, model):
    if model is None:
        return False
    # confirm we analyze models trained with the same p_omission, iti_min, iti_p, etc.
    if 'starkweather' in experiment_name:
        if 'task{}'.format(model['task_index']) not in experiment_name:
            return False
        if 'task2' in experiment_name and model['p_omission_task_2'] != P_OMISSION:
            return False
    elif experiment_name == 'babayan':
        if model.get('reward_time', None) != REWARD_TIME:
            return False
    if model['ncues'] != 1:
        return False
    if model['rnn_mode'] != 'value':
        return False
    if model['gamma'] != GAMMA:
        return False
    if model['iti_p'] != ITI_P:
        return False
    if model['iti_min'] != ITI_MIN:
        return False
    if model['recurrent_cell'] != RECURRENT_CELL:
        return False
    return True

def make_rnn_model(hidden_size):
    return ValueRNN(input_size=2,
        output_size=1,
        hidden_size=hidden_size,
        gamma=GAMMA,
        recurrent_cell=RECURRENT_CELL,
        bias=True, learn_weights=True)

def get_weightsfile(jsonfile, rnn, model_type):
    if 'untrained' in model_type:
        if 'weightsfile_initial' not in rnn:
            return
        else:
            weightsfile = rnn['weightsfile_initial']
    elif 'weightsfile' not in rnn:
        # return
        weightsfile = jsonfile.replace('.json', '.pth')
    else:
        weightsfile = rnn['weightsfile']
    # access these weights files assuming they are in the same dir as the json file
    return os.path.join(os.path.split(jsonfile)[0], os.path.split(weightsfile)[1])

def load_model(jsonfile, model_type, hidden_size):
    assert model_type in ['value-rnn-trained', 'value-rnn-untrained']
    model = json.load(open(jsonfile))
    if hidden_size is not None and model['hidden_size'] != hidden_size:
        return None
    rnn = make_rnn_model(model['hidden_size'])
    weightsfile = get_weightsfile(jsonfile, model, model_type)
    if weightsfile is None:
        if 'untrained' not in model_type:
            return None
    else:
        rnn.load_weights_from_path(weightsfile)
    model['model'] = rnn
    model['weightsfile'] = weightsfile
    model['jsonfile'] = jsonfile
    return model

def get_models(experiment_name, model_type, indir=None, hidden_size=None, esn_gains=ESN_GAINS):
    models = []
    if model_type in ['value-rnn-trained', 'value-rnn-untrained']:
        jsonfiles = get_modelfiles(experiment_name, indir)
        for jsonfile in jsonfiles:
            model = load_model(jsonfile, model_type, hidden_size)
            if rnn_model_is_valid(experiment_name, model):
                models.append(model)
    elif model_type == 'value-esn':
        if hidden_size is None:
            raise Exception("You must provide a hidden_size to analyze for the value-esn")
        for gain in esn_gains:
            rnn = make_rnn_model(hidden_size)
            rnn.initialize(gain)
            model = {}
            model['hidden_size'] = hidden_size
            model['model'] = rnn
            model['gain'] = gain
            models.append(model)
    elif model_type == 'pomdp':
        models.append({})
    else:
        raise Exception("Unrecognized model type: {}".format(model_type))
    for model in models:
        model['experiment_name'] = experiment_name
        model['model_type'] = model_type
        model['gamma'] = GAMMA
    return models

def save_sessions(sessions, args):
    [session.pop('Trials') for session in sessions]
    results = {
        'experiment': args.experiment,
		'model_type': args.model_type,
        'hidden_size': args.hidden_size,
        'sigma': args.sigma,
		'sessions': sessions
		}
    filename = '{}_{}'.format(args.experiment, args.model_type)
    if args.hidden_size:
        filename += '_h{}'.format(args.hidden_size)
    outfile = os.path.join(args.outdir, filename + '.pickle')
    pickle.dump(results, open(outfile, 'wb'))

def main(args):
    experiments = get_experiments(args.experiment)
    models = get_models(args.experiment, args.model_type, args.indir, args.hidden_size)
    print("Found {} valid {} models for experiment {}.".format(len(models), args.model_type, args.experiment))
    pomdp = session.analyze(get_models(args.experiment, 'pomdp')[0], experiments)
    sessions = [session.analyze(model, experiments, pomdp, args.sigma) for model in models]
    save_sessions(sessions, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str,
        choices=['babayan', 'starkweather-task1', 'starkweather-task2'],
        help='which experiment to analyze')
    parser.add_argument('-m', '--model_type', type=str,
        choices=['value-rnn-trained', 'value-rnn-untrained', 'value-esn', 'pomdp'],
        help='which model type to analyze')
    parser.add_argument('--hidden_size', type=int,
        default=None,
        help='hidden size(s) to analyze for rnns (None analyzes all rnns)')
    parser.add_argument('-s', '--sigma', type=float,
        default=0.01,
        help='std dev of noise added to rnn responses')
    parser.add_argument('-i', '--indir', type=str,
        default='valuernn/data',
        help='where to find model files (.json and .pth)')
    parser.add_argument('-o', '--outdir', type=str,
        default='data/sessions',
        help='where to save analysis files (.pickle)')
    args = parser.parse_args()
    main(args)
