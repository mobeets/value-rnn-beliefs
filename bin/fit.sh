#!/bin/bash

N_MODELS=12
N_EPOCHS=150
SAVEDIR='data/models'
DEFAULT_HIDDEN_SIZE=50

for i in 2 5 10 20 50 100
do
	python valuernn/quick_train.py test --experiment starkweather -k $i -t 1 -n $N_MODELS --n_epochs $N_EPOCHS -d $SAVEDIR
	python valuernn/quick_train.py test --experiment starkweather -k $i -t 2 -n $N_MODELS --n_epochs $N_EPOCHS -d $SAVEDIR
done

python valuernn/quick_train.py test --experiment babayan --ntrials_per_episode 50 --reward_time 10 -k $DEFAULT_HIDDEN_SIZE -n $N_MODELS --n_epochs $N_EPOCHS -d $SAVEDIR
