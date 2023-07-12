import os.path
import numpy as np
from plotting.base import plt, colors

def plot_loss(experiment_name, Sessions, outdir):
    hs = [2,5,10,20,50,100]
    for key, items in Sessions.items():
        if key != 'value-rnn-trained':
            continue
        Ls = {}
        for h in hs:
            c = 0
            Ls[h] = []
            for i, rnn in enumerate(items):
                if rnn['hidden_size'] != h:
                    continue
                c += 1
                plt.subplot(3,4,c)
                plt.plot(rnn['scores'], zorder=1)
                plt.xlim([0,250])
                plt.plot(plt.xlim(), [0,0], 'k-', zorder=-1, alpha=0.3)
                if 'starkweather' in experiment_name:
                    plt.plot(plt.xlim(), [0.05,0.05], 'k-', zorder=-1, alpha=0.3)
                plt.plot(plt.xlim(), min(rnn['scores'])*np.ones(2), 'r-', zorder=0, alpha=0.5)
                if 'starkweather' in experiment_name:
                   plt.ylim([-0.002, 0.05])
                Ls[h].append(min(rnn['scores']))
                plt.axis('off')
            plt.tight_layout()
            if len(Ls[h]) > 0:
                plt.savefig(os.path.join(outdir, '{}_loss_{}.pdf'.format(experiment_name, h)))
            plt.close()

            for h, vs in Ls.items():
                if len(vs) == 0:
                    continue
                plt.plot(h*np.ones(len(vs)), vs, 'k.', alpha=0.5)
                plt.plot(h, np.median(vs), 'ko', zorder=-1)
            plt.xlabel('hidden size')
            plt.ylabel('best loss')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, '{}_loss_all.pdf'.format(experiment_name)))
            plt.close()
        return

def get_plotting_info(experiment_name, attr_name, byModelSize=False, hidden_size=None):
    model_names = ['pomdp', 'value-rnn-untrained', 'value-rnn-trained']
    labels = ['Beliefs', 'Untrained RNN', 'Value RNN']
    yticks = []
    if attr_name == 'rpe-mse':
        valgetter = lambda item: item['results']['value']['mse']['rpe_mse']
        ylbl = 'RPE MSE'
        if not byModelSize:
            yl = [-0.0006, 0.01]
            yticks = [0, 0.01]
        else:
            yl = []
            yticks = [0, 0.01]
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
            yl = [-0.8, 0.05]
    elif 'memory-difference' in attr_name:
        o_ngetter = lambda item: len(item['results']['memories']['odor_memories'])
        o_valgetter = lambda item: item['results']['memories']['odor_memories'][0]['duration'] if o_ngetter(item) == 1 else np.nan
        r_ngetter = lambda item: len(item['results']['memories']['rew_memories'])
        r_valgetter = lambda item: item['results']['memories']['rew_memories'][0]['duration'] if r_ngetter(item) == 1 else np.nan
        valgetter = lambda item: o_valgetter(item) - r_valgetter(item)
        ylbl = 'Odor – Rew. Memory'
        yl = []
        if '-with-esn' in attr_name:
            model_names = [('value-esn', hidden_size), 'value-rnn-trained']
            labels = ['Value ESN', 'Value RNN']
        else:
            model_names = model_names[1:]
            labels = labels[1:]
    return model_names, labels, valgetter, ylbl, yl, yticks

def by_model(attr_name, experiment_name, Sessions, outdir, hidden_size, figname, gain_to_plot=None):
    # Figs 3D, 4B-C, 7D-E: plot RPE MSE, belief-rsq, and decoding-LL per model
    model_names, labels, valgetter, ylbl, yl, yticks = get_plotting_info(experiment_name, attr_name, hidden_size=hidden_size)
    plt.figure(figsize=(1.4,2.35))
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
            if key == ('value-esn', hidden_size):
                items = [item for item in items if np.isclose(item['gain'], gain_to_plot)]
        vs = [valgetter(item) for item in items]
        if 'memory' in attr_name:
            thresh = 250
            ignored = [v for v in vs if np.abs(v) > thresh]
            if len(ignored) > 0:
                print('Ignoring {} memory outliers where abs value is > {}: {}'.format(len(ignored), thresh, ignored))
            vs = [v for v in vs if np.abs(v) < thresh]
        mu = np.nanmedian(vs)
        print('{} {} ({}, {:0.2f}: {:0.2f} ± {:0.2f})'.format(experiment_name, key, attr_name, np.nanmedian(vs), np.mean(vs), np.std(vs)/np.sqrt(len(vs))))
        color = colors[key] if len(key) > 2 else colors[key[0]]

        plt.plot(xind, mu, 'o', color=color, alpha=1, zorder=0)
        plt.plot(xind*np.ones(len(vs)) + 0.1*(np.random.rand(len(vs))-0.5), vs, '.',
            markersize=5, color=color, markeredgewidth=0.5, markeredgecolor='k', alpha=1, zorder=1)
    
    plt.xticks(ticks=range(len(model_names)), labels=labels, rotation=90)
    plt.xlim([-0.5, len(model_names)-0.5])
    plt.ylabel(ylbl)
    if yl:
        plt.ylim(yl)
    if yticks:
        plt.yticks(yticks)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def by_model_size(attr_name, experiment_name, Sessions, outdir, figname):
    # Fig 6: plot RPE MSE, belief-rsq, and decoding-LL as a function of model size
    _, _, valgetter, ylbl, yl, yticks = get_plotting_info(experiment_name, attr_name, byModelSize=True)
    # model_names = ['value-rnn-trained']
    model_names = ['value-rnn-untrained', 'value-rnn-trained']

    plt.figure(figsize=(2,2))
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
        mus = np.array([np.nanmedian([v for x,v in zip(xs,vs) if x == xc]) for xc in xsa])
        plt.plot(xsa, mus, 'o', color=color, zorder=0)

        if 'memory' in attr_name:
            thresh = 30
            ignored = [(x,v) for x,v in zip(xs,vs) if np.abs(v) > thresh]
            if len(ignored) > 0:
                print('Ignoring {} memory outliers where abs value is > {}: {}'.format(len(ignored), thresh, ignored))
            try:
                xs, vs = zip(*[(x,v) for x,v in zip(xs,vs) if np.abs(v) < thresh])
            except ValueError:
                xs = []
                vs = []
            plt.plot(plt.xlim(), [0, 0], 'k-', linewidth=1, zorder=-1)

        if 'untrained' in key:
            continue

        plt.plot(xs + 0.0*(np.random.rand(len(vs))-0.5), vs, '.',
            markersize=5, color=color, markeredgewidth=0.5, markeredgecolor='k', alpha=1, zorder=1)
    plt.xlabel('# of units')
    plt.xscale('log')
    plt.ylabel(ylbl)
    if yl:
        plt.ylim(yl)
    if yticks:
        plt.yticks(yticks)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()
