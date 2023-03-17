import os.path
import numpy as np
from plotting.base import plt, colors

def example_time_series(experiment_name, model, outdir):
    plt.figure(figsize=(2.5,2.5))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_trials_{}.pdf'.format(experiment_name)))

def rpes_starkweather(experiment_name, model, outdir, name):
    plt.figure(figsize=(2.5,2.5))

    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_rpes_{}.pdf'.format(experiment_name, name)))

def example_trajectories(experiment_name, model, outdir):
    plt.figure(figsize=(2.5,2.5))

    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_example_trajs.pdf'.format(experiment_name)))

def heatmaps(experiment_name, model, outdir, name):
    plt.figure(figsize=(2.5,2.5))

    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_example_heatmaps_{}.pdf'.format(experiment_name, name)))
