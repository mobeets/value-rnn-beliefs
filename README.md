## Installation

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
chmod +x bin/plot.sh
./bin/plot.sh
```

The resulting figures will then be available at `data/figures/`.
