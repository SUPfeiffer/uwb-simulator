"""Plot Widgets for the UWB Simulation GUI

This file contains several plot widgets that can be used to plot 
simulation data in real time and redraw the plots with matplotlib for
better quality.

Classes:
    QLivePlot: Base class for real time plots
    QLivePlot_Groundtrack: Real time plot for groundtrack
    QLivePlot_Position: Real time plot for x, y, z positions
    QLivePlot_Velocity: Real time plot for x, y, z velocities
    QLivePlot_Attitude: Real time plot for attitude
"""

from PyQt5 import QtWidgets
import pyqtgraph as pg 
import matplotlib.pyplot as plt
import numpy as np

from UWBsim.utils import dataTypes

class QLivePlot(QtWidgets.QWidget):
    """Base Class for real time plots using pyqtgraph
    
    Methods:
        reset: clear the plot area and data
        update_data: Pass new data to the plot widget
        update_plot: Update the plot with the most recent data
    """

    def __init__(self, *args, **kwargs):
        """Initialize the QLivePlot class

        Initializes the widget and creates the basic elements required
        for plotting.
        """

        super(QLivePlot,self).__init__(*args, **kwargs)
        self.layout = QtWidgets.QVBoxLayout()

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground('#FAFAFA')
        self.layout.addWidget(self.canvas)
        
        self.export_button = QtWidgets.QPushButton('plot with matplotlib')
        self.export_button.clicked.connect(self._export_button_clicked)
        self.layout.addWidget(self.export_button)

        self.setLayout(self.layout)

        # same colors used by matplotlib
        #self.data_colors = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#8C564B']
        
        # Switch true and ekf color for publication
        self.data_colors = ['#2CA02C','#FF7F0E','#1F77B4','#D62728','#9467BD','#8C564B']
        
        self.n_subplots = 1
        self.data = [{}]
        self.lines = [{}]

    def reset(self):
        for i in range(self.n_subplots):
            self.plot_area[i].clear()
            self.plot_area[i].legend.items = []
        
        self.data = [{} for _ in range(self.n_subplots)]
        self.lines = [{} for _ in range(self.n_subplots)]

    def update_data(self, **estimates):
        return NotImplemented
    
    def update_plot(self):
        for i in range(self.n_subplots): 
            for key, values in self.data[i].items():
                if key in self.lines[i]:
                    self.lines[i][key].setData(values[0], values[1])
                else:
                    color_i = len(self.lines[i])
                    self.lines[i][key] = self.plot_area[i].plot(values[0], values[1], name=key, pen=self.data_colors[color_i])

    def _export_button_clicked(self):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
        fig = plt.figure()
        ax = []
        for i in range(self.n_subplots):
            ax.append(fig.add_subplot(self.n_subplots,1,i+1))
            legend = []
            for key, values in self.data[i].items():
                ax[i].plot(values[0], values[1])
                legend.append(key)
            ax[i].legend(legend)
            ax[i].grid(b=True)
        plt.show()
        

class QLivePlot_GroundTrack(QLivePlot):
    def __init__(self, *args, **kwargs):
        super(QLivePlot_GroundTrack, self).__init__(*args, **kwargs)

        self.n_subplots = 1
        self.plot_area = []

        self.plot_area.append(self.canvas.addPlot())
        self.plot_area[0].setTitle('Ground Track')
        self.plot_area[0].setLabels(left='y [m]', bottom='x [m]')
        self.plot_area[0].setXRange(-4,4,padding=0)
        self.plot_area[0].setYRange(-4,4,padding=0)
        self.plot_area[0].showGrid(x=True, y=True)
        self.plot_area[0].addLegend()

        self.color_i = 0
    
    def update_data(self, **drone_state_data):
        for key, state in drone_state_data.items():
            if key == 'time':
                continue
            else:
                x = state[0]
                y = state[1]

                if key in self.data[0]:
                    self.data[0][key][0].append(x)
                    self.data[0][key][1].append(y)
                else:
                    self.data[0][key] = [[x],[y]]
                    self.color_i += 1
    

    def _export_button_clicked(self):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
        fig = plt.figure()
        ax = []
        for i in range(self.n_subplots):
            ax.append(fig.add_subplot(self.n_subplots,1,i+1))
            legend = []
            j = 0
            for key, values in self.data[i].items():
                ax[i].plot(values[0], values[1], color=self.data_colors[j])
                legend.append(key)
                j += 1
            #ax[i].legend(legend)
            # Publication legend
            ax[i].legend(["Ground truth","MHE", "EKF"])
            ax[i].grid(b=True)
            ax[i].set_xlabel('x [m]')
            ax[i].set_ylabel('y [m]')
            #ax[i].set_title('Groundtrack')
        plt.show()

class QLivePlot_Position(QLivePlot):
    def __init__(self, *args, **kwargs):
        super(QLivePlot_Position, self).__init__(*args, **kwargs)

        self.n_subplots = 3
        self.data = [{},{},{}]
        self.lines = [{},{},{}]

        self.plot_area = []
        for i in range(self.n_subplots):
            self.plot_area.append(self.canvas.addPlot())
            self.plot_area[i].showGrid(x=True, y=True)
            self.plot_area[i].addLegend()
            self.canvas.nextRow()

        self.plot_area[0].setLabels(left='x [m]', bottom='t [s]')
        self.plot_area[1].setLabels(left='y [m]', bottom='t [s]')
        self.plot_area[2].setLabels(left='z [m]', bottom='t [s]')

        self.color_i = 0

    def update_data(self, **drone_state_data):
        for key,state in drone_state_data.items():
            if key == 'time':
                continue
            else:
                x = state[0]
                y = state[1]
                z = state[2]
                t = drone_state_data['time']

                if key in self.data[0]:
                    self.data[0][key][0].append(t)
                    self.data[0][key][1].append(x)
                    self.data[1][key][0].append(t)
                    self.data[1][key][1].append(y)
                    self.data[2][key][0].append(t)
                    self.data[2][key][1].append(z)
                else:
                    self.data[0][key] = [[t],[x]]
                    self.data[1][key] = [[t],[y]]
                    self.data[2][key] = [[t],[z]]


    def _export_button_clicked(self):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
        ylabels = ['x [m]', 'y [m]', 'z [m]']
        
        fig = plt.figure()
        ax = []
        for i in range(self.n_subplots):
            ax.append(fig.add_subplot(self.n_subplots,1,i+1))
            legend = []
            for key, values in self.data[i].items():
                ax[i].plot(values[0], values[1])
                legend.append(key)
            ax[i].legend(legend)
            ax[i].grid(b=True)
            ax[i].set_xlabel('t [s]')
            ax[i].set_ylabel(ylabels[i])
        plt.show()

class QLivePlot_Velocity(QLivePlot):
    def __init__(self, *args, **kwargs):
        super(QLivePlot_Velocity, self).__init__(*args, **kwargs)

        self.n_subplots = 3
        self.data = [{},{},{}]
        self.lines = [{},{},{}]

        self.plot_area = []
        for i in range(self.n_subplots):
            self.plot_area.append(self.canvas.addPlot())
            self.plot_area[i].showGrid(x=True, y=True)
            self.plot_area[i].addLegend()
            self.canvas.nextRow()

        self.plot_area[0].setLabels(left='vx [m/s]', bottom='t [s]')
        self.plot_area[1].setLabels(left='vy [m/s]', bottom='t [s]')
        self.plot_area[2].setLabels(left='vz [m/s]', bottom='t [s]')

        self.color_i = 0

    def update_data(self, **drone_state_data):
        for key,state in drone_state_data.items():
            if key == 'time':
                continue
            else:
                vx = state[3]
                vy = state[4]
                vz = state[5]
                t = drone_state_data['time']

                if key in self.data[0]:
                    self.data[0][key][0].append(t)
                    self.data[0][key][1].append(vx)
                    self.data[1][key][0].append(t)
                    self.data[1][key][1].append(vy)
                    self.data[2][key][0].append(t)
                    self.data[2][key][1].append(vz)
                else:
                    self.data[0][key] = [[t],[vx]]
                    self.data[1][key] = [[t],[vy]]
                    self.data[2][key] = [[t],[vz]]


class QLivePlot_Attitude(QLivePlot):
    def __init__(self, *args, **kwargs):
        super(QLivePlot_Attitude, self).__init__(*args, **kwargs)

        self.n_subplots = 3
        self.data = [{},{},{}]
        self.lines = [{},{},{}]

        self.plot_area = []
        for i in range(self.n_subplots):
            self.plot_area.append(self.canvas.addPlot())
            self.plot_area[i].showGrid(x=True, y=True)
            self.plot_area[i].addLegend()
            self.canvas.nextRow()

        self.plot_area[0].setLabels(left='Roll [rad]', bottom='t [s]')
        self.plot_area[1].setLabels(left='Pitch [rad]', bottom='t [s]')
        self.plot_area[2].setLabels(left='Yaw [rad]', bottom='t [s]')

        self.color_i = 0

    def update_data(self, **kwargs):
        for key,value in kwargs.items():
            if isinstance(value, dataTypes.State_XVQW):
                r = value.q.get_roll()
                p = value.q.get_pitch()
                y = value.q.get_yaw()
                t = value.timestamp

                if key in self.data[0]:
                    self.data[0][key][0].append(t)
                    self.data[0][key][1].append(r)
                    self.data[1][key][0].append(t)
                    self.data[1][key][1].append(p)
                    self.data[2][key][0].append(t)
                    self.data[2][key][1].append(y)
                else:
                    self.data[0][key] = [[t],[r]]
                    self.data[1][key] = [[t],[p]]
                    self.data[2][key] = [[t],[y]]

            else:
                pass
                #print('Plot can only be updated with State_XVQW data type.')