import os.path
import numpy as np
from plotting.base import plt, colors

esnColor = colors['value-esn']
rnnColor = colors['value-rnn-trained']

def summary_by_gain(attr_name, Sessions, outdir, hidden_size, figname):
    if attr_name == 'odor-memory':
        ngetter = lambda item: len(item['results']['memories']['odor_memories'])
        valgetter = lambda item: item['results']['memories']['odor_memories'][0]['duration'] if ngetter(item) > 0 else np.nan
    elif attr_name == 'reward-memory':
        ngetter = lambda item: len(item['results']['memories']['rew_memories'])
        valgetter = lambda item: item['results']['memories']['rew_memories'][0]['duration'] if ngetter(item) > 0 else np.nan
    elif attr_name == 'belief-rsq':
        valgetter = lambda item: item['results']['belief_regression']['rsq']
    elif attr_name == 'rpe-mse':
        valgetter = lambda item: item['results']['value']['mse']['rpe_mse']
    
    key = ('value-esn', hidden_size)
    if key not in Sessions:
        print("ERROR: Could not find any {} models in processed sessions data.".format(key))
        return
    xs = [item['gain'] for item in Sessions[key]]
    ys = [valgetter(item) for item in Sessions[key]]

    plt.figure(figsize=(2.5,2.5))
    plt.plot(xs, ys, '.', color=esnColor)

    if attr_name in ['rpe-mse', 'belief-rsq']:
        # show average for trained RNNs'
        ysc = [valgetter(item) for item in Sessions.get('value-rnn-trained', []) if item['hidden_size'] == hidden_size]
        if len(ysc) > 0:
            mu = np.nanmean(ysc)
            plt.plot(plt.xlim(), mu*np.ones(2), '--', linewidth=1.5, zorder=-1, color=rnnColor)

    if attr_name == 'odor-memory':
        plt.yticks(ticks=[0,50,100,150,200])
        plt.ylim([0, 220])
        plt.ylabel('Odor memory', fontsize=12)
    elif attr_name == 'reward-memory':
        plt.yticks(ticks=[0,50,100,150,200])
        plt.ylim([0, 220])
        plt.ylabel('Reward memory', fontsize=12)
    elif attr_name == 'belief-rsq':
        plt.ylim([-0.02,1.02])
        plt.ylabel('Belief $R^2$', fontsize=12)
    elif attr_name == 'rpe-mse':
        plt.ylabel('RPE MSE', fontsize=12)
    
    plt.xlabel('Gain', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def activations(valueesns, outdir, figname, xmax=200):
    # Fig 8A-B: plot ESN activations vs time following odor input
    plt.figure(figsize=(5,2.5))
    for i, rnn in enumerate(valueesns):
        plt.subplot(1,2,i+1)
        Ys = rnn['results']['memories']['odor_memories'][0]['trajectory']
        plt.plot(Ys, alpha=0.8)
        plt.xlabel('Time steps rel.\nto odor input', fontsize=12)
        plt.ylabel('Activation', fontsize=12)
        plt.ylim(0.1*np.array([-1,1]))
        plt.xlim([0,xmax])
        plt.yticks([-0.2, 0, 0.2])
        plt.title('ESN, Gain={}'.format(rnn['gain']), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()
