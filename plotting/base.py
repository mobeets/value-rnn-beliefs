import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

colors = {
    'beliefs': 'k',
    'value-rnn-trained': '#6311CE',
    'value-esn': '#CC3432'
}
