import os.path
import numpy as np
from plotting.base import plt, colors

esnColor = colors['value-esn']
rnnColor = colors['value-rnn-trained']

def plot_memories_scatter(results, outfile=None):
    plt.figure(figsize=(2.05,2.05))
    for key, res in results.items():
        xs = res[:,1]
        ys = res[:,0]
        mu = res.mean(axis=0)
        color = colors[key] if key in colors else colors[key[0]]
        plt.plot(xs, ys, '.', markersize=2, alpha=0.5, color=color)
        plt.plot(mu[1], mu[0], 'o', markersize=4, zorder=-1, color=color)

    xmin = np.hstack([plt.xlim(), plt.ylim()]).min()-5
    xmax = np.hstack([plt.xlim(), plt.ylim()]).max()+5
    plt.xlim([xmin, xmax]); plt.ylim(plt.xlim())
    plt.plot([xmin, xmax], [xmin, xmax], 'k-', zorder=-2, linewidth=1)
    plt.xlabel('Reward memory duration')
    plt.ylabel('Odor memory duration')
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()

def memory_comparison(Sessions, hidden_size, gain, outdir, figname):
    keys = [('value-esn', hidden_size), 'value-rnn-trained']
    results = {}
    for key in keys:
        if key not in Sessions:
            print("ERROR: Could not find any {} models in processed sessions data.".format(key))
            return
        results[key] = []
        for item in Sessions[key]:
            if key[0] == 'value-esn' and not np.isclose(item['gain'], gain):
                continue
            if item['hidden_size'] != hidden_size:
                continue
            odor_mems = item['results']['memories']['odor_memories']
            rew_mems = item['results']['memories']['rew_memories']
            if len(odor_mems) != 1 or len(rew_mems) != 1:
                continue
            odor_dur = odor_mems[0]['duration']
            rew_dur = rew_mems[0]['duration']
            results[key].append((odor_dur, rew_dur))
        if len(results[key]) > 0:
            results[key] = np.vstack(results[key])
    plot_memories_scatter(results, os.path.join(outdir, figname + '.pdf'))

def summary_by_gain(attr_name, Sessions, outdir, hidden_size, figname):
    # Fig 8C-E: plot odor memory, RPE MSE, and belief-rsq vs gain for ESNs
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
    elif attr_name == 'state-LL':
        valgetter = lambda item: item['results']['state_decoding']['LL']
    
    key = ('value-esn', hidden_size)
    if key not in Sessions:
        print("ERROR: Could not find any {} models in processed sessions data.".format(key))
        return
    if attr_name in ['odor-memory', 'reward-memory']:
        ymax = 200
        yover = 210
    xs = np.array([item['gain'] for item in Sessions[key]])
    ys = np.array([valgetter(item) for item in Sessions[key]])

    xsa = np.unique(xs)
    mus = np.array([np.nanmedian(ys[xs == x]) for x in xsa])
    xs = xsa; ys = mus

    plt.figure(figsize=(2,2))
    if attr_name in ['odor-memory', 'reward-memory']:
        ix = ys < ymax
        plt.plot(xs[ix], ys[ix], '.', color=esnColor)
        plt.plot(xs[~ix], yover*np.ones((~ix).sum()), '.', color='#ED9F9F')        
    else:
        plt.plot(xs, ys, '.', color=esnColor)
    
    # plt.plot(xsa, mus, '.', markersize=8, color=esnColor, zorder=0)

    # show average for trained RNNs'
    ysc = [valgetter(item) for item in Sessions.get('value-rnn-trained', []) if item['hidden_size'] == hidden_size]
    if len(ysc) > 0:
        mu = np.nanmedian(ysc)
        plt.plot(plt.xlim(), mu*np.ones(2), '--', linewidth=1.5, zorder=-1, color=rnnColor)

    if attr_name == 'odor-memory':
        plt.yticks(ticks=[0,100,200])
        plt.ylim([0, yover+10])
        plt.ylabel('Odor memory')
    elif attr_name == 'reward-memory':
        plt.yticks(ticks=[0,100,200])
        plt.ylim([0, yover+10])
        plt.ylabel('Reward memory')
    elif attr_name == 'belief-rsq':
        plt.ylim([-0.02,1.02])
        plt.yticks([0, 0.5, 1.0])
        plt.ylabel('Belief $R^2$')
    elif attr_name == 'rpe-mse':
        plt.ylabel('RPE MSE')
        plt.yticks([0, 0.01])
        plt.ylim([-0.002, 0.014])
    elif attr_name == 'state-LL':
        plt.ylabel('Log-likelihood')
        plt.yticks([-1, 0])
        plt.ylim([-1.8, 0.05])
    plt.title('Task 2 ESNs')
    
    plt.xlabel('Gain')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def activations(valueesns, outdir, figname, xmax=50):
    # Fig 8A-B: plot ESN activations vs time following odor input
    plt.figure(figsize=(4,2))
    for i, rnn in enumerate(valueesns):
        plt.subplot(1,2,i+1)
        Ys = rnn['results']['memories']['odor_memories'][0]['trajectory']
        plt.plot(Ys, alpha=0.8)
        plt.xlabel('Time steps rel. to odor')
        plt.ylabel('Activation')
        plt.ylim(0.1*np.array([-1,1]))
        plt.xlim([0,xmax])
        plt.yticks([-0.2, 0, 0.2])
        plt.title('ESN, Gain={}'.format(rnn['gain']), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()
