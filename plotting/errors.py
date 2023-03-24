import os.path
import numpy as np
from plotting.base import plt, colors

def get_plotting_info(experiment_name, attr_name, byModelSize=False):
    model_names = ['pomdp', 'value-rnn-untrained', 'value-rnn-trained']
    labels = ['Beliefs', 'Untrained RNN', 'Value RNN']
    if attr_name == 'rpe-mse':
        valgetter = lambda item: item['results']['value']['mse']['rpe_mse']
        ylbl = 'RPE MSE'
        yl = [-0.0006, 0.01]
        model_names = model_names[1:]
        labels = labels[1:]
    elif attr_name == 'belief-rsq':
        valgetter = lambda item: item['results']['belief_regression']['rsq']
        ylbl = 'Belief $R^2$'
        yl = [-0.03, 1.03]
        model_names = model_names[1:]
        labels = labels[1:]
    elif attr_name == 'state-LL':
        valgetter = lambda item: item['results']['state_decoding']['LL']
        ylbl = 'Log-likelihood'
        if 'starkweather' in experiment_name:
            yl = [-0.5, 0.03] if not byModelSize else [-2, 0.03]
        else:
            yl = [-1, 0.03]
    return model_names, labels, valgetter, ylbl, yl

def by_model(attr_name, experiment_name, Sessions, outdir, hidden_size, figname):
    # Figs 3D, 4B-C, 7D-E: plot RPE MSE, belief-rsq, and decoding-LL per model
    model_names, labels, valgetter, ylbl, yl = get_plotting_info(experiment_name, attr_name)
    plt.figure(figsize=(1.8,2.5))
    for xind, key in enumerate(model_names):
        if key not in Sessions:
            print("ERROR: Could not find any {} models in processed sessions data.".format(key))
            continue
        else:
            items = Sessions[key]
            if 'rnn' in key:
                items = [item for item in items if item['hidden_size'] == hidden_size]
                if len(items) == 0:
                    print("ERROR: Could not find any {}, H={} models in processed sessions data.".format(key, hidden_size))
        vs = [valgetter(item) for item in items]
        mu = np.median(vs)
        lb = np.percentile(vs, 25)
        ub = np.percentile(vs, 75)
        print('{} ({}, {:0.2f}: {:0.2f} ± {:0.2f})'.format(experiment_name, attr_name, np.median(vs), np.mean(vs), np.std(vs)/np.sqrt(len(vs))))
        color = colors[key]

        if 'rnn' in key:
            plt.plot(xind*np.ones(2), [lb, ub], '-',
                color=color, linewidth=6, alpha=0.2, zorder=-1)
        plt.plot(xind, mu, 'o', color=color, alpha=1, zorder=0)
        plt.plot(xind*np.ones(len(vs)) + 0.1*(np.random.rand(len(vs))-0.5), vs, '.',
            markersize=5, color=color, markeredgewidth=0.5, markeredgecolor='k', alpha=1, zorder=1)
    
    plt.xticks(ticks=range(len(model_names)), labels=labels, rotation=90, fontsize=12)
    plt.xlim([-0.5, len(model_names)-0.5])
    plt.ylabel(ylbl)
    if yl:
        plt.ylim(yl)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def by_model_size(attr_name, experiment_name, Sessions, outdir, figname):
    # Fig 6: plot RPE MSE, belief-rsq, and decoding-LL as a function of model size
    _, _, valgetter, ylbl, yl = get_plotting_info(experiment_name, attr_name, byModelSize=True)
    model_names = ['value-rnn-trained']

    plt.figure(figsize=(2.5,2.5))
    for key in model_names:
        if key not in Sessions:
            print("ERROR: Could not find any {} models in processed sessions data.".format(key))
            continue
        items = Sessions[key]
        color = colors[key]
        if 'rnn' in key:
            items = [item for item in items]
        vs = [valgetter(item) for item in items]
        if key == 'pomdp':
            mu = np.mean(vs)
            plt.plot(plt.xlim(), mu*np.ones(2), '--', color=color, zorder=-1)
            continue
        xs = [item['hidden_size'] for item in items]
        xsa = np.unique(xs)
        mus = np.array([np.median([v for x,v in zip(xs,vs) if x == xc]) for xc in xsa])
        # sef = lambda xs: np.std(xs)/np.sqrt(len(xs))
        # ses = np.array([sef([v for x,v in zip(xs,vs) if x == xc]) for xc in xsa])
        # lbs = mus-ses; ubs = mus+ses
        lbs = np.array([np.percentile([v for x,v in zip(xs,vs) if x == xc], 25) for xc in xsa])
        ubs = np.array([np.percentile([v for x,v in zip(xs,vs) if x == xc], 75) for xc in xsa])
        plt.plot(xsa, mus, 'o', color=color, zorder=0)
        plt.plot(xs + 0.0*(np.random.rand(len(vs))-0.5), vs, '.',
            markersize=5, color=color, markeredgewidth=0.5, markeredgecolor='k', alpha=1, zorder=1)
        for (xs,lb,ub) in zip(xsa, lbs, ubs):
            plt.plot(xs*np.ones(2), [lb,ub], '-', linewidth=6, color=color, alpha=0.2, zorder=-1)
        # plt.gca().fill_between(xsa, lbs, ubs, linewidth=0, alpha=0.2, color=color)
    plt.xlabel('# of units')
    plt.xscale('log')
    plt.ylabel(ylbl)
    if yl:
        plt.ylim(yl)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()
