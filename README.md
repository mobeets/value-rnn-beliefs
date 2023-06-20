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

Alternatively, to use the fitted models analyzed in the paper, delete the empty `data/models` folder and then unzip `data/models.zip`.

## Analyze and make figures

First, we must analyze all fitted models:

```bash
chmod +x bin/analyze.sh
./bin/analyze.sh
```

Now, we can make the figures:

```bash
chmod +x bin/plot.sh
./bin/plot.sh
```

The resulting figures will then be available at `data/figures/`.

(Note: If you encounter an error on Mac regarding `libomp.dylib`, try `conda install nomkl`.)
