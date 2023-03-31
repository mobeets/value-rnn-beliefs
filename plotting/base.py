import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.figsize'] = [3.0, 3.0]
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

colors = {
    'pomdp': 'k',
    'value-rnn-trained': '#6311CE',
    'value-rnn-untrained': '#B793E5',
    'value-esn': '#CC3432',
    'rewRespSmall': '#0000C4',
    'rewRespBig': '#BB271A',
}
