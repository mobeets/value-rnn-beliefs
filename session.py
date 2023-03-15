import numpy as np
from valuernn.train import make_dataloader, probe_model
import analysis.value
import analysis.correlations
import analysis.decoding
import analysis.dynamics

def add_activity(model, experiment, sigma=0):
	if model['model_type'] == 'pomdp':
		for trial in experiment.trials:
			trial.Z = trial.B
	else:
		dataloader = make_dataloader(experiment, batch_size=1)
		experiment.trials = probe_model(model['model'], dataloader, inactivation_indices=None)
	experiment.trials = experiment.trials[1:]

	# add noise (proportional to std. dev of activity across trials)
	if sigma > 0:
		sds = np.vstack([trial.Z for trial in experiment.trials]).std(axis=0)
		for trial in experiment.trials:
			trial.Z += sigma*sds*np.random.randn(*trial.Z.shape)

def analyze(model, experiments, sigma=0):
	session = {'weights': {}, 'performance': {}}
	for key, val in model.items():
		if key == 'model': # ignore rnn weights
			continue
		assert key not in session
		session[key] = val
	if model['model_type'] != 'pomdp':
		session['sigma'] = sigma
	else:
		session['sigma'] = 0.0

	# add model activity
	for name, experiment in experiments.items():
		add_activity(model, experiment, session['sigma'])
	
	# fit value weights and get rpes
	# analysis.value.fit_value_weights(trials)
	# analysis.value.add_value_and_rpe(trials, value_weights)
	# analysis.value.score_rpe_mse(trials)
	session['weights']['value'] = None
	session['performance']['rpe_mse'] = None

	# fit belief weights
	if model['model_type'] != 'pomdp':
		session['weights']['W_beliefs'] = None
		session['performance']['belief_rsq'] = None

	# fit decoding weights
	session['weights']['W_states'] = None
	session['performance']['state_LL'] = None

	# characterize dynamics
	if model['model_type'] != 'pomdp':
		session['performance']['odor_memory'] = None
		session['performance']['reward_memory'] = None

	return session
