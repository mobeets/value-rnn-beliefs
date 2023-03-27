#!/bin/bash

HIDDEN_SIZE=50
DATADIR='/Users/mobeets/code/value-rnn/data'
# DATADIR='valuernn/data'

# Starkweather Task 1
python analyze.py -e starkweather-task1 -m pomdp -i $DATADIR
python analyze.py -e starkweather-task1 -m value-rnn-trained -i $DATADIR
python analyze.py -e starkweather-task1 -m value-rnn-untrained -i $DATADIR
python plot.py -e starkweather-task1 -i $DATADIR

# Starkweather Task 2
python analyze.py -e starkweather-task2 -m pomdp -i $DATADIR
python analyze.py -e starkweather-task2 -m value-rnn-trained -i $DATADIR
python analyze.py -e starkweather-task2 -m value-rnn-untrained -i $DATADIR
python analyze.py -e starkweather-task2 -m value-esn -i $DATADIR --hidden_size $HIDDEN_SIZE
python plot.py -e starkweather-task2 -i $DATADIR

# Babayan task
python analyze.py -e babayan -m pomdp -i $DATADIR
python analyze.py -e babayan -m value-rnn-trained -i $DATADIR
python analyze.py -e babayan -m value-rnn-untrained -i $DATADIR
python plot.py -e babayan -i $DATADIR
