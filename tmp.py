#%%

from analyze import get_experiments, get_models
import session

#%%

experiment_name = 'starkweather-task2'
model_type = 'value-rnn-trained'
indir = '/Users/mobeets/code/value-rnn/data'
hidden_size = None
experiments = get_experiments(experiment_name)
models = get_models(experiment_name, model_type, indir, hidden_size)
print(len(models))

models = [x for x in models if 'newloss_43866228' in x['weightsfile']]

#%%

pomdp = session.analyze(get_models(experiment_name, 'pomdp')[0], experiments)

#%%

S = session.analyze(models[0], experiments, pomdp, 0.01)

#%%

import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%%

import matplotlib.pyplot as plt
trial = S['Trials']['test'][0]
plt.plot(trial.S)

#%%

print(S['results']['value']['mse']['value_mse'], S['results']['belief_regression']['rsq'], S['results']['state_decoding']['LL'])

#%%

W = S['results']['state_decoding']['state_weights']

#%%

sessions = []
for model in models:
    sessions.append(session.analyze(model, experiments, 0.1))

# %%

