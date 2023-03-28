import numpy as np
import analysis.beliefs_starkweather, analysis.beliefs_babayan

def add_states_and_beliefs(experiment_name, experiment, block_prior=0.5, belief_reward_sigma=0.001, prior_by_prev_block=(1,0)):

    if 'starkweather' in experiment_name:
        T, O = analysis.beliefs_starkweather.pomdp(cue=0, p_omission=experiment.omission_probability, bin_size=experiment.bin_size, ITIhazard=experiment.iti_p, nITI_microstates=experiment.iti_min+1)
        S, observations = analysis.beliefs_starkweather.get_states_and_observations(experiment.trials, cue=0, iti_min=experiment.iti_min)
        B = analysis.beliefs_starkweather.get_beliefs(observations, T, O)
    
    elif 'babayan' in experiment_name:
        assert len(np.unique(experiment.reward_times_per_block)) == 1
        reward_times = experiment.reward_times_per_block[0] + np.arange(-experiment.jitter, experiment.jitter+1)
        T, O = analysis.beliefs_babayan.pomdp(reward_times, p_omission=0.0, ITIhazard=experiment.iti_p, nITI_microstates=experiment.iti_min+1, jitter=experiment.jitter)
        X = np.vstack([x.X for x in experiment.trials])
        reward_amounts = experiment.reward_sizes_per_block if experiment_name == 'babayan' else (experiment.reward_sizes_per_block[0], experiment.reward_sizes_per_block[-1])
        ntrials_per_block = experiment.ntrials_per_block if experiment_name == 'babayan' else None
        B, _ = analysis.beliefs_babayan.get_beliefs(X, T, O,
            prior=block_prior,
            reward_amounts=reward_amounts,
            reward_sigma=belief_reward_sigma,
            prior_by_prev_block=prior_by_prev_block,
            ntrials_per_block=ntrials_per_block)
        if len(experiment.reward_sizes_per_block) == 2:
            S, _ = analysis.beliefs_babayan.get_states_and_observations(experiment.trials,
                reward_amounts=experiment.reward_sizes_per_block,
                iti_min=experiment.iti_min)
        else:
            S = np.nan*np.ones(len(B))
    else:
        raise Exception("Unrecognized experiment for getting beliefs: {}".format(experiment_name))
    
    # add states and beliefs to trials
    i = 0
    for trial in experiment.trials:
        trial.B = B[i:(i+trial.trial_length)]
        trial.S = S[i:(i+trial.trial_length)]
        i += trial.trial_length

    experiment.pomdp = {'T': T, 'O': O}
    return experiment
