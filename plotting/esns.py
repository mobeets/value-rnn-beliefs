import os.path.join
import numpy as np
from plotting.base import plt, colors

esnColor = colors['value-esn']
rnnColor = colors['value-rnn']

def summary_by_gain_attr(attr_name, experiment_name, Sessions, outdir):
    xs = []
    ys = []

    plt.figure(figsize=(2.5,2.5))
    plt.plot(xs, ys, '.', color=esnColor)

    if attr_name in ['rpe-mse', 'belief-rsq']:
        # todo: add in value-rnn avg
        ysc = []
        mu = np.mean(ysc)
        plt.plot(plt.xlim(), mu*np.ones(2), '--', linewidth=1.5, zorder=-1, color=rnnColor)

    if attr_name == 'odor-memory':
        plt.yticks(ticks=[0,50,100,150,200])
        plt.ylim([0, 220])
        plt.ylabel('Odor memory', fontsize=12)
    elif attr_name == 'belief-rsq':
        plt.ylim([-0.02,1.02])
    elif attr_name == 'rpe-mse':
        pass
    
    plt.xlabel('Gain', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_esns_{}.pdf'.format(experiment_name, attr_name)))

def summary_by_gain(experiment_name, Sessions, outdir):
    # Fig 8C-E: plot odor memory, RPE MSE, and belief-rsq vs gain for ESNs
    for attr_name in ['odor-memory', 'rpe-mse', 'belief-rsq']:
        summary_by_gain_attr(attr_name, experiment_name, Sessions, outdir)

def activations(experiment_name, valueesns, outdir, xmax=200):
    # Fig 8A-B: plot ESN activations vs time following odor input
    plt.figure(figsize=(5,2.5))
    for rnn in valueesns:
        plt.subplot(1,2,c+1)
        
        plt.plot(Ts[gain][:,:20], alpha=0.8)

        plt.xlabel('Time rel. to odor input', fontsize=12)
        plt.ylabel('Activation', fontsize=12)
        plt.ylim(0.1*np.array([-1,1]))
        plt.xlim([0,xmax])
        plt.yticks([-0.2, 0, 0.2])
        plt.title('ESN, Gain={}'.format(rnn['gain']), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_esns.pdf'.format(experiment_name)))
