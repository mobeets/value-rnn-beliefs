import os.path
import numpy as np
from plotting.base import plt, colors

beliefColor = colors['pomdp']

def traj(Sessions, outdir, hidden_size, xline, input_name, figname, xtick, experiment_name=None, xmax=50):
    # Figs 5C, 5D: plot distance from ITI following observations, across models
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
