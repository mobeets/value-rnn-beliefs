#!/bin/bash
HIDDEN_SIZE=50 # for ESN only
DATADIR='data/models'

# Starkweather Task 1
python analyze.py -e starkweather-task1 -m pomdp -i $DATADIR
python analyze.py -e starkweather-task1 -m value-rnn-trained -i $DATADIR
python analyze.py -e starkweather-task1 -m value-rnn-untrained -i $DATADIR

# Starkweather Task 2
python analyze.py -e starkweather-task2 -m pomdp -i $DATADIR
python analyze.py -e starkweather-task2 -m value-rnn-trained -i $DATADIR
python analyze.py -e starkweather-task2 -m value-rnn-untrained -i $DATADIR
python analyze.py -e starkweather-task2 -m value-esn -i $DATADIR --hidden_size $HIDDEN_SIZE

# Babayan task
python analyze.py -e babayan -m pomdp -i $DATADIR
python analyze.py -e babayan -m value-rnn-trained -i $DATADIR
python analyze.py -e babayan -m value-rnn-untrained -i $DATADIR
