import os.path
from plotting.base import plt, colors

def by_model(attr_name, experiment_name, Sessions, outdir, hidden_size):
    # Figs 3D, 4B-C, 7D-E: plot RPE MSE, belief-rsq, and decoding-LL per model
    plt.figure(figsize=(2.5,2.5))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_byModel_{}.pdf'.format(experiment_name, attr_name)))

def by_model_size(attr_name, experiment_name, Sessions, outdir):
    # Fig 6: plot RPE MSE, belief-rsq, and decoding-LL as a function of model size
    plt.figure(figsize=(2.5,2.5))

    plt.xlabel('# of units')
    # plt.ylabel(ylbl)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_byModelSize_{}.pdf'.format(experiment_name, attr_name)))
