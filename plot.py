import os.path
import glob
import pickle
import argparse
import numpy as np
import analyze
import session
import plotting.errors, plotting.memories, plotting.esns, plotting.misc

DEFAULT_ISI_MAX = 14 # Starkweather only
DEFAULT_ITI_MIN = 10 # Starkweather only
ESN_GAIN_TO_PLOT = 1.9 # Starkweather-task2 only

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
            sessions = results['sessions']
            if 'rnn' in results['model_type']:
                sessions = [x for x in sessions if x['hidden_size'] in [2,5,10,20,50,100]]
            Sessions[key] = sessions
    return Sessions

def summary_plots(experiment_name, Sessions, outdir, hidden_size, iti_min=DEFAULT_ITI_MIN, isi_max=DEFAULT_ISI_MAX, gain_to_plot=ESN_GAIN_TO_PLOT):
    if experiment_name == 'babayan-interpolate':
        return
    
    # Figs 3D, 4B-C, 7D-E: plot RPE MSE, belief-rsq, and decoding-LL per model
    if 'starkweather' in experiment_name:
        attrnames = ['rpe-mse', 'belief-rsq', 'state-LL', 'memory-difference', 'memory-difference-with-esn']
        fignames = ['Fig3D', 'Fig4B', 'Fig4C', 'Fig5E', 'Fig8H']
        fignames = [x + ('_top' if 'task1' in experiment_name else '_bottom') for x in fignames]
        if 'task1' in experiment_name:
            attrnames = attrnames[:-1]
            fignames = fignames[:-1]
        elif 'task2' in experiment_name:
            fignames[-1] = fignames[-1].replace('_bottom', '')
    else:
        attrnames = ['belief-rsq', 'state-LL']
        fignames = ['Fig7D', 'Fig7E']
    for figname, attr_name in zip(fignames, attrnames):
        plotting.errors.by_model(attr_name, experiment_name, Sessions, outdir, hidden_size, figname=figname, gain_to_plot=gain_to_plot)

    # Summarize number of trained RNNs with each number of fixed points
    nfps = [rnn['results']['memories']['n_fixed_points'] for rnn in Sessions['value-rnn-trained'] if rnn['hidden_size'] == hidden_size]
    if len(nfps) > 0:
        counts = ['{} with {} FP'.format(len([n for n in nfps if n==cn]), cn) for cn in np.unique(nfps)]
        print('Number of fixed points in value-rnn-trained on {}: H={}: {}'.format(experiment_name, hidden_size, ', '.join(counts)))

    if 'babayan' in experiment_name:
        return

    # Figs 5C, 5D: plot distances from ITI following odor/reward, across models
    figname = 'Fig5C_top' if 'task1' in experiment_name else 'Fig5C_bottom'
    plotting.memories.traj(Sessions, outdir, hidden_size, isi_max, 'odor', figname=figname, xtick=DEFAULT_ISI_MAX)
    figname = 'Fig5D_top' if 'task1' in experiment_name else 'Fig5D_bottom'
    plotting.memories.traj(Sessions, outdir, hidden_size, iti_min, 'reward', figname=figname, xtick=DEFAULT_ITI_MIN)

    # figname = 'Fig5E_top' if 'task1' in experiment_name else 'Fig5E_bottom'
    # plotting.memories.memory_difference(Sessions, hidden_size, outdir, figname=figname)
    
    # Fig 6: plot RPE MSE, belief-rsq, and decoding-LL as a function of model size
    if experiment_name == 'starkweather-task2':
        for figname, attr_name in zip(['Fig6A', 'Fig6B', 'Fig6C', 'Fig6D'], ['rpe-mse', 'belief-rsq', 'state-LL', 'memory-difference']):
            plotting.errors.by_model_size(attr_name, experiment_name, Sessions, outdir, figname=figname)

def esn_plots(experiment_name, Sessions, valueesns, outdir, hidden_size, gain_to_plot=ESN_GAIN_TO_PLOT):
    # Fig 8A-B: plot ESN activations vs time following odor input
    plotting.esns.activations(valueesns, outdir, figname='Fig8A-B')

    # Fig 8C-G: plot odor memory, RPE MSE, and belief-rsq vs gain for ESNs
    for figname, attr_name in zip(['Fig8C', 'Fig8D', 'Fig8E', 'Fig8F', 'Fig8G'], ['rpe-mse', 'belief-rsq', 'state-LL', 'odor-memory', 'reward-memory']):
        plotting.esns.summary_by_gain(attr_name, Sessions, outdir, hidden_size, figname=figname)

def single_rnn_plots_starkweather(experiment_name, pomdp, valuernn, untrainedrnn, outdir, iti_min=DEFAULT_ITI_MIN):
    # Fig 2, Fig 4, Fig S1A: plot observations, model activity, value estimate, and RPE on example trials
    if experiment_name == 'starkweather-task2':
        plotting.misc.example_time_series(experiment_name, pomdp, outdir, iti_min, figname='Fig2_beliefs')
        if valuernn is not None:
            plotting.misc.example_time_series(experiment_name, valuernn, outdir, iti_min, figname='Fig2', showPredictions=False)
            plotting.misc.example_time_series(experiment_name, valuernn, outdir, iti_min, figname='Fig4', showValueAndRpes=False)
        if untrainedrnn is not None:
            plotting.misc.example_time_series(experiment_name, untrainedrnn, outdir, iti_min, figname='SuppFig1A',
                showValueAndRpes=False, showPredictions=False)

    # Fig 3B, 3C: plot RPEs as a function of reward time
    figname = 'Fig3B_top' if 'task1' in experiment_name else 'Fig3B_bottom'
    plotting.misc.rpes_starkweather(experiment_name, pomdp, outdir, iti_min, figname=figname)
    if valuernn is not None:
        figname = 'Fig3C_top' if 'task1' in experiment_name else 'Fig3C_bottom'
        plotting.misc.rpes_starkweather(experiment_name, valuernn, outdir, iti_min, figname=figname)
    
    # Fig 5B: plot 2D model activity trajectories on example trials
    if valuernn is not None:
        figname = 'Fig5B_top' if 'task1' in experiment_name else 'Fig5B_bottom'
        plotting.misc.example_trajectories(experiment_name, valuernn, outdir, figname=figname)

    # Fig S1B-C: plot heatmaps of temporal tuning
    if valuernn is not None:
        plotting.misc.heatmaps(valuernn, outdir, figname='SuppFig1B')
    if untrainedrnn is not None:
        plotting.misc.heatmaps(untrainedrnn, outdir, figname='SuppFig1C')

def single_rnn_plots_babayan(experiment_name, pomdp, valuernn, outdir):
    # Fig 7C: plot RPEs as a function of trial index in block
    plotting.misc.rpes_babayan([pomdp, valuernn], outdir, figname='Fig7C')

    # Fig 7G, Fig S3A: plot 2D model activity trajectories on example trials
    if valuernn is not None:
        plotting.misc.example_trajectories(experiment_name, valuernn, outdir, figname='Fig7G', showPretendOmissions=False)
        plotting.misc.example_trajectories(experiment_name, valuernn, outdir, figname='SuppFig3A', showPretendOmissions=True)

    # Fig S3B: plot distance from ITI following odor observation
    if valuernn is not None:
        plotting.misc.example_block_distances(valuernn, outdir, figname='SuppFig3B')

def load_exemplar_models(experiment_name, indir, hidden_size, sigma):
    experiments = analyze.get_experiments(experiment_name)
    pomdp = session.analyze(analyze.get_models(experiment_name, 'pomdp')[0], experiments, doDecode=False)
    if 'babayan' in experiment_name:
        pomdp = session.analyze(analyze.get_models(experiment_name, 'pomdp')[0], experiments, doDecode=False)

    valuernns = analyze.get_models(experiment_name, 'value-rnn-trained', indir, hidden_size)
    if len(valuernns) == 0:
        print("WARNING: Could not find any value-rnn-trained model files (.json).")
        valuernn = None
    else:
        weightsfile = None
        if experiment_name == 'starkweather-task1':
            weightsfile = os.path.join(indir, 'newloss_46377719_501_value_starkweather_task1_gru_h50_itimin10_1cues-v0.pth')
        elif experiment_name == 'starkweather-task2':
            weightsfile = os.path.join(indir, 'newloss_46377799_501_value_starkweather_task2_gru_h50_itimin10_1cues-v0.pth')
        elif 'babayan' in experiment_name:
            weightsfile = os.path.join(indir, 'newloss4_47062592_501_value_babayan_task_gru_h50_itimin10_1cues-v0.pth')
        if weightsfile:
            tmp_valuernns = valuernns
            valuernns = [rnn for rnn in valuernns if os.path.split(rnn['weightsfile'])[-1] == os.path.split(weightsfile)[-1]]
            if len(valuernns) == 0:
                print("WARNING: Could not find exemplar model for {}. Choosing a different one.".format(experiment_name))
                valuernns = tmp_valuernns
        valuernn = session.analyze(valuernns[0], experiments, pomdp, sigma, doDecode=False)
    
    if 'starkweather' in args.experiment:
        untrainedrnns = analyze.get_models(experiment_name, 'value-rnn-untrained', indir, hidden_size)
        if len(untrainedrnns) == 0:
            print("WARNING: Could not find any value-rnn-untrained model files (.json).")
            untrainedrnn = None
        else:
            untrainedrnn = session.analyze(untrainedrnns[0], experiments, pomdp, sigma, doDecode=False)
    else:
        untrainedrnn = None

    if 'task2' in experiment_name:
        valueesns = analyze.get_models(experiment_name, 'value-esn', indir, hidden_size, esn_gains=[0.9, 1.9])
        valueesns = [session.analyze(x, experiments, pomdp, sigma, doDecode=False) for x in valueesns]
    else:
        valueesns = None
    return pomdp, valuernn, untrainedrnn, valueesns

def main(args):
    Sessions = load_sessions(args.experiment, args.sessiondir)
    plotting.errors.plot_loss(args.experiment, Sessions, args.outdir)
    summary_plots(args.experiment, Sessions, args.outdir, args.hidden_size)

    pomdp, valuernn, untrainedrnn, valueesns = load_exemplar_models(args.experiment, args.indir, args.hidden_size, args.sigma)
    if args.experiment == 'babayan':
        single_rnn_plots_babayan(args.experiment, pomdp, valuernn, args.outdir)
    elif args.experiment == 'babayan-interpolate':
        plotting.misc.rpes_babayan_interpolate(Sessions, args.outdir, 'SuppFig2')
    elif 'starkweather' in args.experiment:
        single_rnn_plots_starkweather(args.experiment, pomdp, valuernn, untrainedrnn, args.outdir)
        if valueesns is not None:
            esn_plots(args.experiment, Sessions, valueesns, args.outdir, args.hidden_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str,
        choices=['babayan', 'babayan-interpolate', 'starkweather-task1', 'starkweather-task2'],
        help='which experiment to analyze')
    parser.add_argument('--hidden_size', type=int,
        default=50,
        help='hidden size to use for summarizing rnn results')
    parser.add_argument('--sigma', type=float,
        default=0.01,
        help='std dev of noise added to rnn responses')
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
