import os.path
import glob
import pickle
import argparse
import analyze
import session
import plotting.errors, plotting.memories, plotting.esns

DEFAULT_ISI_MAX = 15 # Starkweather only
DEFAULT_ITI_MIN = 10 # Starkweather only
DEFAULT_HIDDEN_SIZE = 50
DEFAULT_SIGMA = 0.01

def load_sessions(experiment_name, sessiondir):
    template = '{}_*'.format(experiment_name)
    filenames = glob.glob(os.path.join(sessiondir, template + '.pickle'))
    Sessions = {}
    for filename in filenames:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
            assert results['experiment'] == experiment_name
            key = results['model_type']
            if results['hidden_size'] is not None:
                key = (key, results['hidden_size'])
            Sessions[key] = results['sessions']
    return Sessions

def summary_plots(experiment_name, Sessions, outdir, hidden_size=DEFAULT_HIDDEN_SIZE, iti_min=DEFAULT_ITI_MIN, isi_max=DEFAULT_ISI_MAX):
    # Figs 3D, 4B-C, 7D-E: plot RPE MSE, belief-rsq, and decoding-LL per model
    for attr_name in ['rpe-mse', 'belief-rsq', 'state-LL']:
        plotting.errors.by_model(attr_name, experiment_name, Sessions, outdir, hidden_size)

    if experiment_name == 'babayan':
        return

    # Fig 6: plot RPE MSE, belief-rsq, and decoding-LL as a function of model size
    for attr_name in ['rpe-mse', 'belief-rsq', 'state-LL']:
        plotting.errors.by_model_size(attr_name, experiment_name, Sessions, outdir)

    # Figs 5C, S2A: plot dsitances from ITI following odor/reward, across models
    plotting.memories.traj(experiment_name, Sessions, outdir, hidden_size, isi_max, 'odor')
    plotting.memories.traj(experiment_name, Sessions, outdir, hidden_size, iti_min, 'reward')

    # Figs 5D, S2B: plot histogram of odor/reward memories, across models
    plotting.memories.histogram(experiment_name, Sessions, outdir, hidden_size, isi_max, 'odor')
    plotting.memories.histogram(experiment_name, Sessions, outdir, hidden_size, iti_min, 'reward')

def esn_plots(experiment_name, Sessions, valueesns, outdir, hidden_size=DEFAULT_HIDDEN_SIZE):
    # Fig 8A-B: plot ESN activations vs time following odor input
    plotting.esns.activations(experiment_name, valueesns, outdir)

    # Fig 8C-E: plot odor memory, RPE MSE, and belief-rsq vs gain for ESNs
    for attr_name in ['odor-memory', 'rpe-mse', 'belief-rsq']:
        plotting.esns.summary_by_gain(attr_name, experiment_name, Sessions, outdir, hidden_size)

def single_rnn_plots_starkweather(experiment_name, pomdp, valuernn, outdir):
    # Fig 2, Fig 4, Fig S1: plot observations, model activity, value estimate, and RPE on example trials
    # Fig 3C: plot RPEs as a function of reward time
    # Fig 5B, Fig 7G, Fig S3A: plot 2D model activity trajectories on example trials
    # Fig S1B-C: plot heatmaps of temporal tuning
    pass

def single_rnn_plots_babayan(experiment_name, pomdp, valuernn, outdir):
    # Fig 7C: plot RPEs as a function of trial index in block
    # Fig 7G, Fig S3A: plot 2D model activity trajectories on example trials
    # Fig S3B: plot distance from ITI following odor observation
    pass

def main(args):
    Sessions = load_sessions(args.experiment, args.sessiondir)
    summary_plots(args.experiment, Sessions, args.outdir)
    return

    experiments = analyze.get_experiments(args.experiment)
    pomdp = session.analyze(analyze.get_models(args.experiment, 'pomdp')[0], experiments)
    valuernn = analyze.get_models(args.experiment, 'value-rnn-trained', args.indir, DEFAULT_HIDDEN_SIZE)[0]
    valuernn = session.analyze(valuernn, experiments, pomdp, DEFAULT_SIGMA)

    if args.experiment == 'babayan':
        single_rnn_plots_babayan(args.experiment, pomdp, valuernn, args.outdir)
    elif 'starkweather' in args.experiment:
        single_rnn_plots_starkweather(args.experiment, pomdp, valuernn, args.outdir)
        if 'task2' in args.experiment:
            valueesns = analyze.get_models(args.experiment, 'value-esn', args.indir, DEFAULT_HIDDEN_SIZE, esn_gains=[0.7, 1.7])
            valueesns = [session.analyze(x, experiments, pomdp, DEFAULT_SIGMA) for x in valueesns]
            esn_plots(args.experiment, Sessions, valueesns, args.outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str,
        choices=['babayan', 'starkweather-task1', 'starkweather-task2'],
        help='which experiment to analyze')
    parser.add_argument('-s', '--sessiondir', type=str,
        default='data/sessions',
        help='where to find session files (.pickle)')
    parser.add_argument('-i', '--indir', type=str,
        default='valuernn/data',
        help='where to find model files (.json and .pth)')
    parser.add_argument('-o', '--outdir', type=str,
        default='data/figures',
        help='where to save figure files (.pdf)')
    args = parser.parse_args()
    main(args)
