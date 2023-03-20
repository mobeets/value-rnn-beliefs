import os.path
import numpy as np
from sklearn.decomposition import PCA
from plotting.base import plt, colors

def example_time_series(experiment_name, model, outdir, iti_min, inds=np.arange(46)+97, nUnitsToShow=20, showObservations=True, showBeliefs=True, showPredictions=True, showValueAndRpes=True):
    
    name = model['model_type']
    trials = model['Trials']['test']
    color = colors[name]
    height = 1.5
    if showPredictions: height += 0.5
    if showValueAndRpes: height += 2
    plt.figure(figsize=(5, height))

    # plot odor and reward observations
    if showObservations:
        pad = 2
        X = np.vstack([t.X for t in trials])
        for c in [0,1]:
            ts = np.where(X[inds,c] > 0)[0]
            plt.plot(c/2 + np.zeros(len(inds)) + pad, 'k-', linewidth=1)
            for t in ts:
                plt.plot([t, t], np.array([c, c+1])/2 + pad, 'k-', linewidth=1)
            pad -= 1.3

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
    plt.savefig(os.path.join(outdir, '{}_trials_{}.pdf'.format(experiment_name, name)))

def rpes_starkweather(experiment_name, model, outdir, iti_min):
    plt.figure(figsize=(1.6, 1.8))
    name = model['model_type']

    xl = (np.array([-0.9, 3.8])/0.2).tolist()
    val_getter_pre = lambda item: item['results']['value']['rpe_summary']['rpes_cue']
    val_getter = lambda item: list(item['results']['value']['rpe_summary']['rpes_mu'].items())
    val_getter_post = lambda item: item['results']['value']['rpe_summary']['rpes_end']
    xlbl = 'Time (s)'
    ylbl = 'RPE'
    yl = [-0.3, 0.75] if experiment_name == 'starkweather' else [-0.5, 1.6]
    
    vs_pre = val_getter_pre(model)
    vs = val_getter(model)
    vs_post = val_getter_post(model)
    v_post = vs_post[0]
    
    xs_pre = np.arange(len(vs_pre)) - iti_min + 2
    xs = vs[:,0]; vs = vs[:,1]

    plt.plot(xs_pre, vs_pre, 'k-', linewidth=1)

    clrs = plt.cm.cool(np.linspace(0,1,xs.max()-xs.min()+1))
    clrs = clrs[::-1]

    for i in range(len(xs)):
        xsc = [xs[i]-1, xs[i], xs[i]+1]
        ysc = [vs_pre[xs_pre == xs[i]-1][0], vs[i], v_post]
        plt.plot(xsc, ysc, '-', linewidth=1, color=clrs[i], alpha=1)

    plt.xlabel(xlbl, fontsize=12)
    plt.ylabel(ylbl, fontsize=12)
    plt.xticks(ticks=xs_pre[3::5], labels=(xs_pre[3::5]*0.2).astype(int), fontsize=12)
    plt.yticks(fontsize=12)
    plt.yticks([])
    if yl:
        plt.ylim(yl)
    if xl:
        plt.xlim(xl)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_rpes_{}.pdf'.format(experiment_name, name)))

def example_trajectories(experiment_name, model, outdir):
    plt.figure(figsize=(2.5,2.5))
    color = colors[model['model_type']]
    nullColor = 'r'
    odorResp = 'k'
    rewResp = 'k'
    nullResp = 'k'
    nullRespOmission = 'c'
    showOmissions = True
    trialIndsToShow = [0] if experiment_name == 'starkweather' else [0,4]

    Z = []
    Z_postRew = []
    ctrials = []
    fpsc = []
    Trajs = []
    TrajsOdor = []
    block_ids = []
    fp_block_ids = []

    pca = PCA(n_components=Z.shape[1])
    pca.fit(Z)
    Zpc = pca.transform(Z)

    xind = 0; yind = 1
    plt.figure(figsize=(2,2))
    zpc = pca.transform(Z_postRew)
    fpc = pca.transform(fpsc)

    for t in trialIndsToShow:
        trial = ctrials[t]
        if trial.Z.shape[1] > model.hidden_size:
            zs = trial.Z[:,:-1]
        else:
            zs = trial.Z
        zs = pca.transform(zs)
        
        plt.plot(zs[trial.iti-1:trial.iti+1,xind], zs[trial.iti-1:trial.iti+1,yind], '-', color=odorResp, alpha=1, markersize=3, zorder=1)
        curColor = nullResp if ctrials[t].y.sum() > 0 else nullRespOmission
        if trial.y.sum() > 0:
            plt.plot(zs[trial.iti+trial.isi-1:trial.iti+trial.isi+1,xind], zs[trial.iti+trial.isi-1:trial.iti+trial.isi+1,yind], '-', color=rewResp, alpha=1, markersize=2, zorder=1)
            zorder = -1
        else:
            zorder = -2
        plt.plot(zs[trial.iti:trial.iti+trial.isi,xind], zs[trial.iti:trial.iti+trial.isi,yind], '.', color=curColor, alpha=1, markersize=2, zorder=zorder)
        plt.plot(zs[trial.iti:trial.iti+trial.isi,xind], zs[trial.iti:trial.iti+trial.isi,yind], '-', color=curColor, alpha=0.5, markersize=2, zorder=zorder)

    for t in trialIndsToShow:
        traj = Trajs[t]
        zs = pca.transform(traj)
        curColor = nullResp if ctrials[t].y.sum() > 0 else nullRespOmission
        plt.plot(zs[:,xind], zs[:,yind], '.', color=curColor, alpha=1, markersize=3, zorder=0)
        plt.plot(zs[:,xind], zs[:,yind], '-', color=curColor, alpha=0.5, markersize=3, zorder=-1)

        if not showOmissions:
            continue
        try:
            traj = TrajsOdor[t]
        except:
            continue
        zs = pca.transform(traj)
        curColor = nullRespOmission
        plt.plot(zs[ctrials[t].isi-1:,xind], zs[ctrials[t].isi-1:,yind], '.', color=curColor, alpha=1, markersize=3, zorder=-1)
        plt.plot(zs[ctrials[t].isi-1:,xind], zs[ctrials[t].isi-1:,yind], '-', color=curColor, alpha=0.5, markersize=3, zorder=-2)

    for b in np.unique(block_ids):
        color = '#6311CE'
        plt.plot(fpc[fp_block_ids==b,xind], fpc[fp_block_ids==b,yind], '.',
                markersize=14, alpha=0.1, color=color,
                markeredgewidth=0.5, markeredgecolor='k')

    plt.xlabel('$z_{}$'.format(xind+1))
    plt.ylabel('$z_{}$'.format(yind+1))
    zmn = Zpc.min(axis=0)-0.4
    zmx = Zpc.max(axis=0)+0.1
    plt.xlim([zmn[xind], zmx[xind]])
    plt.ylim([zmn[yind], zmx[yind]])
    plt.xticks([]); plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '{}_example_trajs.pdf'.format(experiment_name)))

def heatmaps(experiment_name, model, outdir):
    responses = model['Trials']['test']
    name = model['model_type']

    sortThisData = True
    iti = 11 # iti duration
    isi = 14 # reward time
    tPre = 2 # number of time steps shown before odor
    tPost = 10 # number of time steps shown after reward

    trialinds = [i for i,x in enumerate(responses) if x.iti==iti and x.isi==isi and x.y.sum() > 0]
    X_hats = []
    for i in trialinds:
        z = responses[i].Z
        znext = responses[i+1].Z
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
    plt.xticks(ticks=[tPre,tPre+isi], labels=['Odor', 'Reward'], fontsize=10)
    plt.yticks(ticks=[], labels=[])
    plt.xlabel('Time$\\rightarrow$', fontsize=12)
    plt.ylabel('Units', fontsize=12)
    plt.tight_layout()
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    plt.savefig(os.path.join(outdir, '{}_example_heatmaps_{}.pdf'.format(experiment_name, name)))
