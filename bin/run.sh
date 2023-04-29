#!/bin/bash

HIDDEN_SIZE=50
DATADIR='data/models'
DO_ANALYZE=true

# Starkweather Task 1
# if $DO_ANALYZE
# then
# 	python analyze.py -e starkweather-task1 -m pomdp -i $DATADIR
# 	python analyze.py -e starkweather-task1 -m value-rnn-trained -i $DATADIR
# 	python analyze.py -e starkweather-task1 -m value-rnn-untrained -i $DATADIR
# fi
# python plot.py -e starkweather-task1 -i $DATADIR

# # Starkweather Task 2
if $DO_ANALYZE
then
	python analyze.py -e starkweather-task2 -m pomdp -i $DATADIR
	python analyze.py -e starkweather-task2 -m value-rnn-trained -i $DATADIR
	# python analyze.py -e starkweather-task2 -m value-rnn-untrained -i $DATADIR
	# python analyze.py -e starkweather-task2 -m value-esn -i $DATADIR --hidden_size $HIDDEN_SIZE
fi
python plot.py -e starkweather-task2 -i $DATADIR

# Babayan task
# if $DO_ANALYZE
# then
# 	python analyze.py -e babayan -m pomdp -i $DATADIR
# 	python analyze.py -e babayan -m value-rnn-trained -i $DATADIR
# 	python analyze.py -e babayan -m value-rnn-untrained -i $DATADIR
# fi
# python plot.py -e babayan -i $DATADIR
