#!/bin/bash
DATADIR='data/models'

# Starkweather Task 1
python plot.py -e starkweather-task1 -i $DATADIR

# Starkweather Task 2
python plot.py -e starkweather-task2 -i $DATADIR

# Babayan task
python plot.py -e babayan -i $DATADIR
python plot.py -e babayan-interpolate -i $DATADIR
