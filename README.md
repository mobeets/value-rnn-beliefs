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

## Fitting models

```bash
chmod +x bin/fit.sh
./bin/fit.sh
```

Approximate run time: 36 hours.

## Analyze and make figures

```bash
chmod +x bin/run.sh
./bin/run.sh
```

The resulting figures will then be available at `data/figures/`.
