import numpy as np
import torch

#%% ODOR/REWARD MEMORY

NSAMPLES = 10 # no. of samples to begin search for fixed point
MAXREPS = 5000 # no. of steps to try to find each fixed point
TOL = 1e-5 # tolerance for converging to fixed point
TOL_MEMORY = 1e-3 # tolerance for measuring memory duration
TOL_UNIQUE = 1e-3 # tolerance for considering fixed points identical to one another

def sample_starting_points(trials, nsamples):
    # sample starting points
    starting_points = {}
    Z = np.vstack([trial.Z for trial in trials])
    xs = np.hstack([trial.X[:,0] for trial in trials])
    
    # activity in response to an odor is an ISI
    ISIs = Z[np.where(xs == 1)[0]]
    if nsamples < len(ISIs):
        ix2 = np.argsort(np.random.rand(len(ISIs)))[:nsamples]
        ISIs = ISIs[ix2]
    starting_points['ISI'] = ISIs

    # activity in response to a reward is an ITI (considers each nonzero reward)
    rs = np.vstack([trial.y for trial in trials])[:,0]
    all_rews = np.unique(rs[rs > 0])
    for r in all_rews:
        ITIs = Z[np.where(rs == r)[0]]
        if nsamples < len(ITIs):
            ix1 = np.argsort(np.random.rand(len(ITIs)))[:nsamples]
            ITIs = ITIs[ix1]
        starting_points['ITI_{}'.format(r)] = ITIs
    return starting_points

def uniquetol(fps, tol_unique):
    if len(fps) == 0: return fps
    ufps = [fps[0]]
    for fp in fps[1:]:
        if np.min([np.linalg.norm(fp - ufp) for ufp in ufps]) > tol_unique:
            ufps.append(fp)
    return ufps

def unsqueeze(x):
    return torch.unsqueeze(torch.unsqueeze(x, 0), 0)

def get_const_input(input_size, key):
    const_signal = torch.Tensor([0]*input_size)
    if 'odor' in key:
        const_signal[0] = 1
    elif 'reward' in key:
        const_signal[1] = 1
    elif 'null' in key:
        pass
    else:
        raise Exception("Did not recognize key: {}".format(key))
    return unsqueeze(const_signal)

def find_fixed_points(rnn, trials, nsamples=NSAMPLES, maxreps=MAXREPS, tol=TOL, tol_unique=TOL_UNIQUE):
    starting_points = sample_starting_points(trials, nsamples)
    null_signal = get_const_input(rnn.input_size, 'null')
    fps = []
    with torch.no_grad():
        for _, hs in starting_points.items():
            for h in hs:
                fp = None
                h = unsqueeze(torch.Tensor(h))
                for _ in range(maxreps):
                    hnext = rnn.rnn(null_signal, h)[0]
                    if torch.norm(h - hnext) < tol:
                        fp = np.squeeze(hnext.detach().numpy())
                        break
                    else:
                        h = hnext
                if fp is not None:
                    fps.append(fp)
    return uniquetol(fps, tol_unique)

def add_memory_trajectory_and_duration(rnn, h0, input_type, maxreps=MAXREPS, tol_memory=TOL_MEMORY, minreps=0):
    null_signal = get_const_input(rnn.input_size, 'null')
    input_signal = get_const_input(rnn.input_size, input_type)
    
    with torch.no_grad():
        traj = [h0]
        h0 = torch.Tensor(h0)
        h = rnn.rnn(input_signal, unsqueeze(h0))[0]
        traj.append(np.squeeze(h.detach().numpy()))
        for i in range(maxreps):
            h = rnn.rnn(null_signal, h)[0]
            if torch.norm(h - h0) < tol_memory and i >= minreps:
                break
            else:
                traj.append(np.squeeze(h.detach().numpy()))
    ds = np.vstack([np.linalg.norm(h-traj[0]) for h in traj[1:]])[:,0]
    return {'distances': ds, 'trajectory': traj, 'duration': len(traj), 'input_type': input_type}

def analyze(model, Trials):
    trials = Trials['test'] # we do not use training trials for these analyses
    fixed_points = find_fixed_points(model['model'], trials)
    n_fixed_points = len(fixed_points)
    if n_fixed_points != 1:
        print("WARNING: Found {} fixed points in model {}".format(n_fixed_points, model.get('weightsfile', model['model_type'])))

    odor_memories = []
    rew_memories = []
    for fixed_point in fixed_points:
        odor_memories.append(add_memory_trajectory_and_duration(model['model'], fixed_point, 'odor'))
        rew_memories.append(add_memory_trajectory_and_duration(model['model'], fixed_point, 'reward'))
    res = {'fixed_points': fixed_points, 'n_fixed_points': n_fixed_points, 'odor_memories': odor_memories, 'rew_memories': rew_memories}
    
    # find trajectories on omission trials
    starting_points = sample_starting_points(trials[:10], 10)
    omission_trials = []
    for starting_point in starting_points['ISI']:
        omission_trials.append(add_memory_trajectory_and_duration(model['model'], starting_point, 'null', minreps=50))
    res['pretend_omission_trials'] = omission_trials
    return res
