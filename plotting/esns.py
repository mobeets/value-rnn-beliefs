import os.path
import numpy as np
from plotting.base import plt, colors

esnColor = colors['value-esn']
rnnColor = colors['value-rnn-trained']

def summary_by_gain(attr_name, experiment_name, Sessions, outdir, hidden_size):
    if attr_name == 'odor-memory':
        valgetter = lambda item: item['results']['memories']['odor_memories'][0]['duration']
    elif attr_name == 'belief-rsq':
        valgetter = lambda item: item['results']['belief_regression']['rsq']
    elif attr_name == 'rpe-mse':
        valgetter = lambda item: item['results']['value']['mse']['rpe_mse']
    
    key = ('value-esn', hidden_size)
    xs = [item['gain'] for item in Sessions[key]]
    ys = [valgetter(item) for item in Sessions[key]]

    plt.figure(figsize=(2.5,2.5))
    plt.plot(xs, ys, '.', color=esnColor)

    if attr_name in ['rpe-mse', 'belief-rsq']:
        # todo: add in value-rnn avg
        ysc = [valgetter(item) for item in Sessions[('value-rnn-trained', hidden_size)]]
        mu = np.mean(ysc)
        plt.plot(plt.xlim(), mu*np.ones(2), '--', linewidth=1.5, zorder=-1, color=rnnColor)

    if attr_name == 'odor-memory':
        plt.yticks(ticks=[0,50,100,150,200])
        plt.ylim([0, 220])
        plt.ylabel('Odor memory', fontsize=12)
    elif attr_name == 'belief-rsq':
        plt.ylim([-0.02,1.02])
        plt.ylabel('Belief $R^2$', fontsize=12)
    elif attr_name == 'rpe-mse':
        plt.ylabel('RPE MSE', fontsize=12)
    
    plt.xlabel('Gain', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_esns_{}.pdf'.format(experiment_name, attr_name)))

def activations(experiment_name, valueesns, outdir, xmax=200):
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
    plt.savefig(os.path.join(outdir, '{}_esn_activations.pdf'.format(experiment_name)))
