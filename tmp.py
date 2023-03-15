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

from valuernn.train import make_dataloader, probe_model

E = experiments['train']
dataloader = make_dataloader(E, batch_size=1)
trials = probe_model(models[0]['model'], dataloader, inactivation_indices=None)
# plt.subplot(1,2,1)
# plt.plot(E.trials[0].Z)
# plt.subplot(1,2,2)
# plt.plot(trials[0].Z)

#%%

sessions = []
for model in models:
    sessions.append(session.analyze(model, experiments, 0.1))
# %%

