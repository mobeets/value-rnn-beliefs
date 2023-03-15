import analysis.rpes
import analysis.correlations
import analysis.decoding
import analysis.dynamics

def add_activity(model, experiment, sigma=0):
	return

def analyze(model, experiments, sigma=0):
	session = {'params': {}, 'weights': {}, 'performance': {}}
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
	for name, experiment in zip(experiments.items()):
		add_activity(model, experiment, session['sigma'])
	
	# fit value weights and get rpes

	# fit belief weights
	if model['model_type'] != 'pomdp':
		pass

	# fit decoding weights

	# characterize dynamics
	if model['model_type'] != 'pomdp':
		pass

	return session
