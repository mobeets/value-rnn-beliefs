import os.path
import glob
import json
import argparse
import numpy as np

import tasks
from model import ValueRNN
import analysis.beliefs
import session

RECURRENT_CELL = 'GRU'
TRAIN_SEED = 555
TEST_SEED = 666
ITI_P = 1/8
ITI_MIN = 10
GAMMA = 0.93
P_OMISSION = 0.1 # starkweather only
REWARD_TIME = 10 # babayan only

def get_experiment(name, seed=None):
	if seed is not None:
		np.random.seed(seed)
	
	if 'starkweather' in name:
		E = tasks.Starkweather(ncues=1,
			ntrials_per_cue=1000,
			ntrials_per_episode=1000,
			omission_probability=P_OMISSION if 'task2' in name else 0.0,
			iti_p=ITI_P, iti_min=ITI_MIN, t_padding=0,
			omission_trials_have_duration=True,
			include_null_input=False,
			half_reward_times=False)
	elif name == 'babayan':
		E = tasks.Babayan(nblocks=(100,100), # was 2*(1000,) but want to match starkweather
			ntrials_per_block=(5,5),
			reward_sizes_per_block=(1,10),
			reward_times_per_block=(5,5),
			jitter=1,#(1 if name == 'train' else 0),
			iti_min=ITI_MIN, iti_p=ITI_P,
			include_unique_rewards=False,
			ntrials_per_episode=None) # n.b. defaults to len(trials)
	else:
		raise Exception("Unrecognized experiment: {}".format(name))
	return E

def get_experiments(name):
	expt_tr = analysis.beliefs.add_states_and_beliefs(name, get_experiment(name, TRAIN_SEED))
	expt_te = analysis.beliefs.add_states_and_beliefs(name, get_experiment(name, TEST_SEED))
	return {'train': expt_tr, 'test': expt_te}

def get_modelfiles(experiment_name, indir, hidden_size=None):
	if 'starkweather' in experiment_name:
		model_name_templates = ['newloss_*']
		ignore_templates = ['*_initial*', '*babayan*']
		if 'task1' in experiment_name:
			ignore_templates.append('*task2*')
		elif 'task2' in experiment_name:
			ignore_templates.append('*task1*')
	elif experiment_name == 'babayan':
		model_name_templates = ['newloss_*babayan*']
		ignore_templates = ['*_initial*', '*starkweather*']
	if hidden_size is not None:
		ignore_templates.append('*h{}*'.format(hidden_size))
	
	for model_name_template in  model_name_templates:
		modelfiles = glob.glob(os.path.join(indir, model_name_template + '.json'))
		for ignore_template in ignore_templates:
			modelfiles = list(set(modelfiles) - set(glob.glob(os.path.join(indir, ignore_template + '.json'))))
	return modelfiles

def rnn_is_valid(experiment_name, rnn):
	# confirm we analyze models trained with the same p_omission, iti_min, iti_p, etc.
	if 'starkweather' in experiment_name:
		if 'task{}'.format(rnn['task_index']) not in experiment_name:
			return False
		if 'task2' in experiment_name and rnn['p_omission_task2'] != P_OMISSION:
			return False
	elif experiment_name == 'babayan':
		if rnn['reward_time'] != REWARD_TIME:
			return False
	if rnn['ncues'] != 1:
		return False
	if rnn['rnn_mode'] != 'value':
		return False
	if rnn['gamma'] != GAMMA:
		return False
	if rnn['iti_p'] != ITI_P:
		return False
	if rnn['iti_min'] != ITI_MIN:
		return False
	if rnn['t_padding'] > 0:
		return False
	if rnn['recurrent_cell'] != RECURRENT_CELL:
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
		weightsfile = rnn['initial_weightsfile']
	else:
		weightsfile = rnn['weightsfile']
	# access these weights files locally, i.e., in same dir as the json file
	return os.path.join(os.path.split(jsonfile)[0], os.path.split(weightsfile)[1])

def load_model(jsonfile, model_type):
	assert model_type in ['value-rnn-trained', 'value-rnn-untrained']
	rnn = json.load(open(jsonfile))
	model = make_rnn_model(rnn)
	weightsfile = get_weightsfile(jsonfile, rnn, model_type)
	model.load_weights_from_path(weightsfile)
	rnn['model'] = model
	rnn['weightsfile'] = weightsfile
	rnn['jsonfile'] = jsonfile
	return rnn

def get_models(experiment_name, model_type, indir, hidden_size=None):
	models = []
	if model_type in ['value-rnn-trained', 'value-rnn-untrained']:
		jsonfiles = get_modelfiles(experiment_name, indir)
		for jsonfile in jsonfiles:
			rnn = load_model(jsonfile, model_type)
			if rnn_is_valid(experiment_name, rnn):
				models.append(rnn)
	elif model_type == 'value-esn':
		gains = np.arange(0.1, 2.8, 0.2)
		if hidden_size is None:
			raise Exception("You must provide a hidden_size to analyze for the value-esn")
		for gain in gains:
			model = make_rnn_model(hidden_size)
			model.initialize(gain)
			rnn['hidden_size'] = hidden_size
			rnn['model'] = model
			rnn['gain'] = gain
			models.append(rnn)
	elif model_type == 'pomdp':
		models.append({})
	else:
		raise Exception("Unrecognized model type: {}".format(model_type))
	for model in models:
		model['experiment_name'] = experiment_name
		model['model_type'] = model_type
	return models

def save_sessions(sessions, args):
	name = '{}_{}'.format(args.experiment_name, args.model_type)
	if args.hidden_size is not None:
		name += '_h{}'.format(args.hidden_size)
	outfile = os.path.join(args.outdir, name  + '.json')
	json.dump(sessions, open(outfile, 'w'))

def main(args):
	experiments = get_experiments(args.experiment_name)
	models = get_models(args.experiment_name, args.model_type, args.indir, args.hidden_size)
	return
	sessions = []
	for model in models:
		sessions.append(session.analyze(model, experiments, args.sigma))
	save_sessions(sessions, args)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--experiment', type=str,
        choices=['babayan', 'starkweather-task1', 'starkweather-task2'],
        help='which experiment to analyze')
	parser.add_argument('-m', '--model_type', type=str,
        choices=['value-rnn-trained', 'value-rnn-untrained', 'value-esn', 'pomdp'],
        help='which model type to analyze')
	parser.add_argument('-h', '--hidden_size', type=int,
		default=None,
        help='hidden size(s) to analyze for rnns (None analyzes all rnns)')
	parser.add_argument('-s', '--sigma', type=float,
		default=0.1,
        help='std dev of noise added to rnn responses')
	parser.add_argument('-i', '--indir', type=str,
		default='data/models',
        help='where to find model files (.json and .pth)')
	parser.add_argument('-o', '--outdir', type=str,
		default='data/sessions',
        help='where to save analysis files')
	args = parser.parse_args()
	main(args)
