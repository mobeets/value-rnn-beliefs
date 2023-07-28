import os.path
import numpy as np
from sklearn.decomposition import PCA
from plotting.base import plt, colors

def example_time_series(experiment_name, model, outdir, iti_min, figname,
                        inds=np.arange(46)+97, nUnitsToShow=20, showObservations=True, showBeliefs=True, showPredictions=True, showValueAndRpes=True):
    
    name = model['model_type']
    trials = model['Trials']['test']
    color = colors[name]
    height = 1.9
    if showPredictions: height += 0.5
    if showValueAndRpes: height += 2
    plt.figure(figsize=(4.2, height))

    # plot odor and reward observations
    if showObservations:
        pad = 2
        X = np.vstack([t.X for t in trials])
        for c in [0,1]:
            ts = np.where(X[inds,c] > 0)[0]
            plt.plot(c/2.5 + np.zeros(len(inds)) + pad, 'k-', linewidth=1)
            for t in ts:
                plt.plot([t, t], np.array([c, c+1])/2.5 + pad, 'k-', linewidth=1)
            pad -= 1.1

    # plot states
    S = np.hstack([trial.S for trial in trials])
    S[S >= S.max()-iti_min] = S.max()-iti_min
    rew_times = np.unique([trial.isi for trial in trials if trial.y.max() > 0])
    rew_clrs = plt.cm.cool(np.linspace(0,1,rew_times.max()-rew_times.min()+1))
    rew_clrs = rew_clrs[::-1]
    clrs = np.vstack([np.zeros((rew_times.min()-1,4)), rew_clrs, np.zeros((S.max()-rew_times.max()+1,4))])
    clrs[:rew_times.min()-1] = clrs[rew_times.min()]
    clrs[rew_times.max():,-1] = 1.
    for c in range(S.max()+1):
        ts = np.where(S[inds] == c)[0]
        for t in ts:
            plt.plot([t, t], np.array([0-1, 0]) + 0.7, '-', linewidth=1, color=clrs[c])

    # plot beliefs
    if showBeliefs:
        Zb = np.vstack([t.B for t in trials])
        for d in reversed(range(Zb.shape[1])):
            plt.plot(Zb[inds,d] - 2.2, color=clrs[d] if d < len(clrs) else 'k', linewidth=1)
    
    # plot activity
    if name != 'pomdp':
        Zr = np.vstack([t.Z for t in trials])
        Zr = (Zr - Zr.min(axis=0))/(Zr.max(axis=0) - Zr.min(axis=0)) # normalize for plotting
        plt.plot(Zr[inds,:nUnitsToShow] - 4.0, linewidth=1)

    # plot belief prediction
    if showPredictions and name != 'pomdp':
        Bhat = np.vstack([t.Bhat for t in trials])
        for d in reversed(range(Bhat.shape[1])):
            plt.plot(Bhat[inds,d] - 5.7, color=clrs[d] if d < len(clrs) else clrs[-1], linewidth=1)
        rsq = model['results']['belief_regression']['rsq']
        print('Belief r-squared ({}, {}): {:0.2f}'.format(experiment_name, name, rsq))

    # plot value and rpes
    if showValueAndRpes:
        V = np.hstack([t.value for t in trials])
        plt.plot(0*V[inds] - 7, '-', linewidth=1, color=0.8*np.ones(3), markersize=1)
        plt.plot(V[inds] - 7, '-', linewidth=1, color=color, markersize=1)
        rpe = np.hstack([t.rpe for t in trials])
        plt.plot(0*V[inds] - 8.2, '-', linewidth=1, color=0.8*np.ones(3), markersize=1)
        plt.plot(rpe[inds] - 8.2, '-', color=color, linewidth=1, markersize=1)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def rpes_babayan(models, outdir, figname):
    for alignType in ['CS', 'US']:
        plt.figure(figsize=(1.7,2))
        for model in models:
            if model is None:
                continue
            name = model['model_type']
            marker = '-' if 'rnn' in name else '--'
            alpha = 0.9 if 'rnn' in name else 0.7
            zorder = 1 if 'rnn' in name else 2
            rpes = model['results']['value']['rpe_summary']
            crpes = rpes[alignType]
            for group in [1,2,3,4]:
                if group == 1:
                    color = '#0000C4'
                elif group == 2:
                    color = '#5297F8'
                elif group == 3:
                    color = '#EF8733'
                else:
                    color = '#BB271A'
                vals = crpes[group]
                ts = vals['times']
                mus = vals['mus']
                ses = vals['ses']
                h = plt.plot(ts, mus, marker, color=color, markersize=10, alpha=alpha, zorder=zorder)
                for t,mu,se in zip(ts, mus, ses):
                    plt.plot(t*np.ones(2), [mu-se, mu+se], '-', color=h[0].get_color(), zorder=zorder)
        plt.xticks(ts)
        plt.xlabel('Trial')
        plt.ylabel('RPE')
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, figname + '_{}.pdf'.format(alignType)))
        plt.close()

def rpes_babayan_interpolate(Sessions, outdir, figname):
    plt.figure(figsize=(4,2))
    reward_sizes_per_block = []
    for c,ti in enumerate([1,2]):
        plt.subplot(1,2,c+1)

        for model_type in ['pomdp', 'value-rnn-trained']:
            models = Sessions[model_type]
            color = colors[model_type]
            alpha = 0.2 if 'rnn' in model_type else 1
            yss = []
            for model in models:
                rpes = model['results']['value']['rpe_summary']
                reward_sizes_per_block = np.unique([r for (_,r) in rpes.keys()])
                ys = [rpes[(ti,r)] for r in reward_sizes_per_block]
                yss.append(ys)
            yss = np.vstack(yss)
            mus = np.mean(yss, axis=0)
            ses = np.std(yss, axis=0)/np.sqrt(yss.shape[0])
            lbs = mus-ses; ubs = mus+ses
            plt.plot(reward_sizes_per_block, mus, '.', markersize=8,
                    color=color, label=model_type,
                    zorder=1 if 'rnn' in model_type else 0)
            for r,lb,ub in zip(reward_sizes_per_block,lbs,ubs):
                plt.plot(r*np.ones(2), [lb,ub], '-', color=color, zorder=-1)

        plt.xlabel('Reward size')
        plt.ylabel('RPE')
        if len(reward_sizes_per_block) < 7:
            plt.xticks(reward_sizes_per_block)
        else:
            plt.xticks([1, 5, 10])
        plt.title('Trial {} in block'.format(ti), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def rpes_starkweather(experiment_name, model, outdir, iti_min, figname):
    plt.figure(figsize=(1.6, 1.8))
    name = model['model_type']

    xl = (np.array([-0.9, 3.8])/0.2).tolist()
    val_getter_pre = lambda item: item['results']['value']['rpe_summary']['rpes_cue']
    val_getter = lambda item: np.vstack(list(item['results']['value']['rpe_summary']['rpes_mu'].items()))
    val_getter_post = lambda item: item['results']['value']['rpe_summary']['rpes_end']
    xlbl = 'Time (s)'
    ylbl = 'RPE'
    yl = [-0.25, 0.55] if 'starkweather' in experiment_name else [-0.5, 1.6]
    
    vs_pre = val_getter_pre(model)
    vs = val_getter(model)
    vs_post = val_getter_post(model)
    v_post = vs_post[0]
    
    vs_pre = np.hstack([vs_pre, vs_post])
    xs_pre = np.arange(len(vs_pre)) - iti_min + 2
    xs = vs[:,0]; vs = vs[:,1]

    plt.plot(xs_pre, vs_pre, 'k-', linewidth=1)

    clrs = plt.cm.cool(np.linspace(0,1,int(xs.max()-xs.min()+1)))
    clrs = clrs[::-1]

    for i in range(len(xs)):
        xsc = [xs[i]-1, xs[i], xs[i]+1]
        ysc = [vs_pre[xs_pre == xs[i]-1][0], vs[i], v_post]
        plt.plot(xsc, ysc, '-', linewidth=1, color=clrs[i], alpha=1)

    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.xticks(ticks=xs_pre[3::5], labels=(xs_pre[3::5]*0.2).astype(int))
    plt.yticks([])
    if yl:
        plt.ylim(yl)
    if xl:
        plt.xlim(xl)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def example_trajectories(experiment_name, model, outdir, figname, showPretendOmissions=True):
    name = model['model_type']
    trials = model['Trials']['test']
    Z = np.vstack([trial.Z for trial in trials])

    odorResp = 'k'
    if 'starkweather' in experiment_name:
        rewColor = colors['rewRespBig']
    else:
        rewRespSmall = colors['rewRespSmall']
        rewRespBig = colors['rewRespBig']
    nullResp = 'k'
    nullRespOmission = 'c'
    trialIndsToShow = [0] if 'starkweather' in experiment_name else [2,7]
    # Trajs = [traj['trajectory'] for traj in model['results']['memories']['pretend_omission_trials']]
    Trajs = model['results']['memories']['pretend_omission_trials']

    pca = PCA(n_components=Z.shape[1])
    pca.fit(Z)
    if showPretendOmissions: # we only want to print this once
        print('Variance explained by first two PCs during {}: {:0.2f}%'.format(experiment_name, 100*pca.explained_variance_ratio_[:2].sum()))

    xind = 0; yind = 1
    plt.figure(figsize=(2,2))

    for t in trialIndsToShow:
        trial = trials[t]
        zs = pca.transform(trial.Z)
        
        # plot odor response
        plt.plot(zs[trial.iti-1:trial.iti+1,xind], zs[trial.iti-1:trial.iti+1,yind],
                '-', color=odorResp, alpha=1, markersize=3, zorder=1)
        
        if trial.y.sum() > 0:
            # plot reward response
            if 'babayan' in experiment_name:
                rewColor = rewRespSmall if trial.y.sum() < 5 else rewRespBig
            plt.plot(zs[trial.iti+trial.isi-1:trial.iti+trial.isi+1,xind], zs[trial.iti+trial.isi-1:trial.iti+trial.isi+1,yind],
                    '-', color=rewColor, alpha=1, markersize=2, zorder=1)
            zorder = -1
        else:
            zorder = -2

        # plot ISI responses
        curColor = nullResp if trial.y.sum() > 0 else nullRespOmission
        plt.plot(zs[trial.iti:trial.iti+trial.isi,xind], zs[trial.iti:trial.iti+trial.isi,yind],
                '.', color=curColor, alpha=1, markersize=2, zorder=zorder)
        plt.plot(zs[trial.iti:trial.iti+trial.isi,xind], zs[trial.iti:trial.iti+trial.isi,yind],
                '-', color=curColor, alpha=0.5, markersize=2, zorder=zorder)
        
        # plot ITI response (post-reward)
        next_trial = trials[t+1]
        zs_next = pca.transform(next_trial.Z)
        zsc = np.vstack([zs[-1:], zs_next[:next_trial.iti-1]])
        plt.plot(zsc[:,xind], zsc[:,yind],
                '.', color=curColor, alpha=1, markersize=2, zorder=zorder)
        plt.plot(zsc[:,xind], zsc[:,yind],
                '-', color=curColor, alpha=0.5, markersize=2, zorder=zorder)
        
        # plot putative fixed point
        plt.plot(zsc[-1,xind], zsc[-1,yind], '.',
                markersize=14, color=colors[name],
                markeredgewidth=0.5, markeredgecolor='k')
        
    # plot omission trials
    if showPretendOmissions:
        for t in trialIndsToShow:
            trial = trials[t]
            traj = Trajs[t]
            zs = pca.transform(traj)
            plt.plot(zs[(trial.isi-1):,xind], zs[(trial.isi-1):,yind],
                    '.', color=nullRespOmission, markersize=3, zorder=-1)
            plt.plot(zs[(trial.isi-1):,xind], zs[(trial.isi-1):,yind],
                    '-', color=nullRespOmission, alpha=0.5, markersize=3, zorder=-2)

    plt.xlabel('$z_{}$'.format(xind+1))
    plt.ylabel('$z_{}$'.format(yind+1))
    Zpc = pca.transform(Z)
    zmn = Zpc.min(axis=0)-0.4
    zmx = Zpc.max(axis=0)+0.1
    plt.xlim([zmn[xind], zmx[xind]])
    plt.ylim([zmn[yind], zmx[yind]])
    plt.xticks([]); plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def example_block_distances(model, outdir, figname):
    plt.figure(figsize=(2,2))

    trials = model['Trials']['test']
    max_iti = 65
    fps = model['results']['memories']['fixed_points']
    if len(fps) != 1:
        print("ERROR: Did not find one fixed point.")
    trajs = [traj['trajectory'] for traj in model['results']['memories']['pretend_omission_trials']]
    for trial, traj in zip(trials, trajs):
        if trial.rel_trial_index == 0:
            continue
        dists = np.sum((traj - fps[0])**2,axis=1)
        color = '#BB271A' if trial.block_index == 1 else '#0000C4'

        plt.plot(dists, '-', alpha=0.7, color=color)
    plt.xlim([-8, 400])
    plt.xticks([0, max_iti, 200])
    plt.xlabel('Time rel. to odor')
    plt.ylabel('Dist. to fixed point')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()

def heatmaps(model, outdir, figname):
    trials = model['Trials']['test']
    name = model['model_type']

    sortThisData = True
    iti = 11 # iti duration
    isi = 14 # reward time
    tPre = 2 # number of time steps shown before odor
    tPost = 10 # number of time steps shown after reward

    trialinds = [i for i,x in enumerate(trials) if x.iti==iti and x.isi==isi and x.y.sum() > 0]
    X_hats = []
    for i in trialinds:
        z = trials[i].Z
        znext = trials[i+1].Z
        zcur = np.vstack([z, znext])
        zcur = zcur[(iti-tPre):(iti+isi+tPost),:-1]
        X_hats.append(zcur)
    X_hats = np.dstack(X_hats)

    # split into train/test, and then average separately (to cross-validate)
    ixTrain = np.argsort(np.random.rand(X_hats.shape[-1])) < 0.5*X_hats.shape[-1]
    Ztr = X_hats[:,:,ixTrain].mean(axis=-1).T
    Zte = X_hats[:,:,~ixTrain].mean(axis=-1).T

    Zd = []
    tmax = []
    for i,(ztr,zte) in enumerate(zip(Ztr, Zte)):
        tmax.append(np.argmax(ztr))
        zc = (zte-zte.min())/(zte.max()-zte.min()) # only affects visualization
        Zd.append(zc)
    if sortThisData:
        ixs = np.argsort(tmax)[::-1]
    else:
        print("Using indices from previously sorted data")
    Zd = np.vstack(Zd)[ixs]

    plt.figure(figsize=(2.7,2.5))
    plt.imshow(Zd, aspect='auto')
    plt.xticks(ticks=[tPre,tPre+isi], labels=['Odor', 'Reward'])
    plt.yticks(ticks=[], labels=[])
    plt.xlabel('Time$\\rightarrow$')
    plt.ylabel('Units')
    plt.tight_layout()
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    plt.savefig(os.path.join(outdir, figname + '.pdf'))
    plt.close()
