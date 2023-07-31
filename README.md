## Installation

First, clone this repo. Next, you will also need to pull the `valuernn` submodule by running:

```bash
git submodule update --init
```

Next, set up the required python packages by creating a virtual environment:

```bash
conda create --name valuernn python=3.9 pytorch matplotlib numpy scipy scikit-learn
conda activate valuernn
```

## Train models

__Option 1.__ To use the models from the paper, unzip `data/models.zip` to the folder `data/models`.

__Option 2.__ To fit your own models (approximate run time: 48 hours), run:
```bash
chmod +x bin/fit.sh
./bin/fit.sh
```

## Analyze models

__Option 1.__ To use the analyses from the paper, unzip `data/sessions.zip` to the folder `data/sessions`.

__Option 2.__ To analyze your own models, run:
```bash
chmod +x bin/analyze.sh
./bin/analyze.sh
```

## Make figures

To make the figures, run:
```bash
chmod +x bin/plot.sh
./bin/plot.sh
```

The resulting figures will be in `data/figures/`.

(Note: If you encounter an error on Mac regarding `libomp.dylib`, try `conda install nomkl`.)
