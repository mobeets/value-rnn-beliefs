import os.path
import numpy as np
from analyze import *
from plotting.base import plt, colors

def plot(args):
    ynms = ['RPE MSE', 'Belief $R^2$', 'Log-likelihood', ]
    yrngs = [[-0.001, 0.019], [-0.03, 1.03], [-2, 0.03]]
    plt.figure(figsize=(6,2))
    for model_type in ['value-rnn-trained', 'value-rnn-untrained']:
        fnm = os.path.join(args.outdir, '{}_{}.npy'.format(args.experiment, model_type))
        df = np.load(fnm)
        df = df[:,[0,1,2,4,3]]
        if args.hidden_size:
            df = df[df[:,1] == args.hidden_size]

        color = colors[model_type]
        for c, ynm, yrng in zip(range(3), ynms, yrngs):
            plt.subplot(1,3,c+1)
            sigma = df[:,0]
            sigmas = np.unique(sigma)
            hs = df[:,1]
            mus = np.array([np.median(df[sigma==s,c+2]) for s in sigmas])
            plt.plot(sigmas, mus, 'o', color=color, zorder=0)

            vss = np.array([df[sigma==s,c+2] for s in sigmas])
            if ynm == 'RPE MSE':
                plt.yticks([0, 0.01])
            if 'untrained' not in model_type:
                for xs,vs in zip(sigmas,vss):
                    plt.plot(xs + 0.0*(np.random.rand(len(vs))-0.5), vs, '.', markersize=5, color=color, markeredgewidth=0.5, markeredgecolor='k', alpha=1, zorder=1)

            plt.xlabel('Noise gain')
            plt.ylabel(ynm)
            plt.xscale('log')
            plt.ylim(yrng)
    plt.tight_layout()
    plt.savefig(os.path.join(args.plotdir, 'SuppFig4.pdf'))
    plt.close()

def analyze(args, model_type):
    experiments = get_experiments(args.experiment)
    models = get_models(args.experiment, model_type, args.indir, args.hidden_size)
    print("Found {} valid {} models for experiment {}.".format(len(models), model_type, args.experiment))
    pomdp = session.analyze(get_models(args.experiment, 'pomdp')[0], experiments)

    pts = []
    models = models
    for sigma in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]:
        items = [session.analyze(model, experiments, pomdp, sigma, noSigmaForPomdp=False) for model in models]
        keygetter = lambda item: item['hidden_size']
        valgetter1 = lambda item: item['results']['value']['mse']['rpe_mse']
        valgetter2 = lambda item: item['results']['state_decoding']['LL']
        valgetter3 = lambda item: item['results']['belief_regression']['rsq']
        valgetter = lambda item: [valgetter1(item), valgetter2(item), valgetter3(item)]
        for item in items:
            x = keygetter(item)
            ys = valgetter(item)
            pts.append((sigma, x, *ys))
    pts = np.vstack(pts)
    fnm = os.path.join(args.outdir, '{}_{}.npy'.format(args.experiment, model_type))
    np.save(fnm, pts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str,
        choices=['starkweather-task1', 'starkweather-task2'],
        default='starkweather-task2',
        help='which experiment to analyze')
    parser.add_argument('--hidden_size', type=int,
        default=50,
        help='hidden size to analyze for rnns (None analyzes all rnns)')
    parser.add_argument('-i', '--indir', type=str,
        default='data/models',
        help='where to find model files (.json and .pth)')
    parser.add_argument('-o', '--outdir', type=str,
        default='data/sessions',
        help='where to save analysis files (.pickle)')
    parser.add_argument('-p', '--plotdir', type=str,
        default='data/figures',
        help='where to save figures (.pdf)')
    args = parser.parse_args()
    analyze(args, 'value-rnn-trained')
    analyze(args, 'value-rnn-untrained')
    plot(args)
