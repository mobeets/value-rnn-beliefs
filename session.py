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
		trials = probe_model(model['model'], dataloader, inactivation_indices=None)
	trials = trials[1:]

	# add noise (proportional to std. dev of activity across trials)
	if sigma > 0:
		sds = np.vstack([trial.Z for trial in trials]).std(axis=0)
		for trial in trials:
			trial.Z += sigma*sds*np.random.randn(*trial.Z.shape)
	
	return trials

def analyze(model, experiments, pomdp=None, sigma=0):
	session = dict((key, val) for key,val in model.items() if key != 'model')
	session['sigma'] = sigma if model['model_type'] != 'pomdp' else 0.0

	# add model activity
	Trials = {}
	for name, experiment in experiments.items():
		Trials[name] = get_activity(model, experiment, session['sigma'])
	session['Trials'] = Trials
	
	# fit value weights and get rpes
	session['value'] = analysis.value.analyze(Trials, gamma=model['gamma'], pomdp=pomdp)

	return session

	# fit belief weights
	if model['model_type'] != 'pomdp':
		session['belief_regression'] = analysis.correlations.analyze(model, Trials, pomdp=pomdp)

	# fit decoding weights
	session['state_decoding'] = analysis.decoding.analyze(model, Trials, pomdp=pomdp)

	# characterize dynamics
	if model['model_type'] != 'pomdp':
		session['memories'] = analysis.dynamics.analyze(model, Trials)

	return session
