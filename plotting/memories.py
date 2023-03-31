import os.path
import numpy as np
from plotting.base import plt, colors

beliefColor = colors['pomdp']

def traj(Sessions, outdir, hidden_size, xline, input_name, figname, xtick, xmax=50):
    # Figs 5C, S2A: plot distance from ITI following observations, across models
    rnns = [rnn for rnn in Sessions.get('value-rnn-trained', []) if rnn['hidden_size'] == hidden_size]
    if len(rnns) == 0:
        print("ERROR: Could not find any value-rnn-trained, H={} models in processed sessions data.".format(hidden_size))
        return

    plt.figure(figsize=(2,2))
    keyname = input_name if input_name == 'odor' else 'rew'

    # plot beliefs
    pomdp = Sessions['pomdp'][0]
    color = colors['pomdp']
    ds = np.array(pomdp['results']['memories']['{}_memories'.format(keyname)])
    plt.plot(ds/ds.max(), alpha=0.8, linewidth=2, color=color, zorder=1)

    # plot rnns
    color = colors['value-rnn-trained']
    for i, rnn in enumerate(rnns):
        mems = rnn['results']['memories']['{}_memories'.format(keyname)]
        if len(mems) != 1:
            continue
        ds = mems[0]['distances']
        plt.plot(ds/ds.max(), alpha=0.8, zorder=0)

    plt.xlabel('Time steps rel. to {}'.format(input_name))
    plt.ylabel('Rel. distance from ITI')
    plt.ylim([0, 1.01])
    plt.xlim([0, xmax])
    plt.xticks([0, xtick, xmax])
    plt.yticks([0, 0.5, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def histogram(experiment_name, Sessions, outdir, hidden_size, xline, input_name, figname, xmax=200):
    # Figs 5D, S2B: plot histogram of odor/reward memories
    rnns = [rnn for rnn in Sessions.get('value-rnn-trained', []) if rnn['hidden_size'] == hidden_size]
    if len(rnns) == 0:
        print("ERROR: Could not find any value-rnn-trained, H={} models in processed sessions data.".format(hidden_size))
        return

    plt.figure(figsize=(2,2))
    bins = np.linspace(0, xmax, 20)
    color = colors['value-rnn-trained']

    keyname = input_name if input_name == 'odor' else 'rew'
    vs = []
    for rnn in rnns:
        mems = rnn['results']['memories']['{}_memories'.format(keyname)]
        if len(mems) != 1:
            continue
        vs.append(mems[0]['duration'])
    if len(vs) > 0:
        ys, xs = np.histogram(vs, bins=bins)
        print('{} ({} memory, {}: {:0.2f} ± {:0.2f})'.format(experiment_name, input_name, np.median(vs), np.mean(vs), np.std(vs)/np.sqrt(len(vs))))

        xs = [np.mean([xs[i], xs[i+1]]) for i in range(len(xs)-1)]
        width = np.diff(xs[:-1]).mean()
        plt.bar(xs, 100*ys/len(vs), width=width, color=color)
        plt.bar(xmax-width/2, len(vs)-ys.sum(), width=width, alpha=1.0, color=color)
    plt.plot(xline*np.ones(2), [0, 100], '--', linewidth=1, color=beliefColor)
    # plt.bar(xline, 100, width=width, color=beliefColor, alpha=0.8)
    
    lblname = input_name if input_name == 'odor' else 'rew.'
    plt.xlabel('{} memory (time steps)'.format(lblname.capitalize()))
    plt.ylabel('% of RNNs')
    plt.xlim([-np.diff(xs).mean(), xmax])
    plt.ylim([0, 100])
    plt.yticks(np.arange(0,101,25))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()
