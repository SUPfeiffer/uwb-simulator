import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib import style

class PlotData:
    def __init__(self, subplot_lines, subplot_columns):
        self.fig = plt.figure()
        n = subplot_lines*subplot_columns
        self.ax = [self.fig.add_subplot(subplot_lines, subplot_columns, i+1) for i in range(n)]
        
        self.reset(n)

    def reset(self, subplots):
        self.x = [[[]] for _ in range(subplots)]
        self.y = [[[]] for _ in range(subplots)]
        self.data_series = [{} for _ in range(subplots)]
        self.titles = [None for _ in range(subplots)]
        self.legends = [[] for _ in range(subplots)]
        self.ylabels = [None for _ in range(subplots)]
        self.xlabels = [None for _ in range(subplots)]

    def config_subplot(self, i, title=None, legend=None, xlabel=None, ylabel=None):
        self.titles[i-1] = title
        self.legends[i-1] = legend
        self.xlabels[i-1] = xlabel
        self.ylabels[i-1] = ylabel

        if legend is not None:
            for lg in legend:
                self.data_series[i-1][lg] = len(self.data_series[i-1])
                self.x[i-1].append([])
                self.y[i-1].append([])
        else:
            self.legends[i-1] = []

    def add_data(self, subplot, series, x, y):
        if not series in self.data_series[subplot-1]:
            self.data_series[subplot-1][series] = len(self.data_series[subplot-1])
            self.x[subplot-1].append([])
            self.y[subplot-1].append([])
            self.legends[subplot-1].append(series)

        s = self.data_series[subplot-1][series]
        self.x[subplot-1][s].append(x)
        self.y[subplot-1][s].append(y)
    
    def plot(self):
        for i,sp in enumerate(self.ax):
            series = len(self.x[i])
            for ser in range(series):
                sp.plot(self.x[i][ser], self.y[i][ser])
            sp.grid(b=True)    
            
            if self.titles[i] is not None:
                sp.set_title(self.titles[i])
            if self.legends[i] is not None:
                sp.legend(self.legends[i])
            if self.xlabels[i] is not None:
                sp.set_xlabel(self.xlabels[i])
            if self.ylabels[i] is not None:
                sp.set_ylabel(self.ylabels[i])
   