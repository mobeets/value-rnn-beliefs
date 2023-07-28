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

for i in 0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 1.9 2.1 2.3 2.5 2.7
do
	python valuernn/quick_train.py valueesn --experiment starkweather -k $DEFAULT_HIDDEN_SIZE -t 2 -n $N_MODELS --n_epochs 0 -d $SAVEDIR --initialization_gain $i
done

python valuernn/quick_train.py test --experiment babayan --ntrials_per_episode 50 --reward_time 10 -k $DEFAULT_HIDDEN_SIZE -n $N_MODELS --n_epochs $N_EPOCHS -d $SAVEDIR
