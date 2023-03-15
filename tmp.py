#%%

from analyze import get_experiments, get_models
import session

#%%

experiment_name = 'starkweather-task1'
model_type = 'value-rnn-trained'
indir = '/Users/mobeets/code/value-rnn/data'
hidden_size = None
experiments = get_experiments(experiment_name)
models = get_models(experiment_name, model_type, indir, hidden_size)
print(len(models))

#%%

sessions = []
for model in models:
    sessions.append(session.analyze(model, experiments, 0.1))
