"""Window to show live status of the current simulation

The simulation window shows live plots and error calculation and handles
the current simulation

Classes:
    SimulationWindow: The GUI window that shows the simulation
    QLiveInfo: PyQT Widget which shows simulation time and errors
    WorkerSignals: Signals for communication between threads
    SimulationWorker: Runs simulation on a separate thread
"""

from PyQt5 import QtWidgets
from PyQt5 import QtCore
import time
import math
import numpy as np
import ptvsd

from UWBsim.interface import plot_widgets
from UWBsim.simulation import SimulationParams, UWBSimulation
from UWBsim.airframe.drone import Drone

class SimulationWindow(QtWidgets.QWidget):
    """GUI window that handles and shows the current simulation

    The SimulationWindow handles the simulation in the GUI and provides
    live information on it's progress with live plots and error calculation

    Usage: Redefine the function get_sim_params() to return the parameter
    structure that is defined in another window.
    """

    default_params = SimulationParams()

    def __init__(self, *args, **kwargs):
        super(SimulationWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('Simulation')
        
        self.threadpool = QtCore.QThreadPool()
        self.sim_stop_flag = False
        
        self.layout = QtWidgets.QGridLayout()

        ## plot tabs
        self.plotTabs = QtWidgets.QTabWidget()
        
        self.plot_list = {}
        self.plot_list['Ground Track'] = plot_widgets.QLivePlot_GroundTrack()
        self.plot_list['Position'] = plot_widgets.QLivePlot_Position()
        self.plot_list['Velocity'] = plot_widgets.QLivePlot_Velocity()

        for key, value in self.plot_list.items():
            self.plotTabs.addTab(value, key)    
        self.layout.addWidget(self.plotTabs,0,0,4,1)

        self.error_wdg = QLiveInfo()
        self.layout.addWidget(self.error_wdg, 0,1)
        
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.clicked.connect(self.run_button_clicked)
        self.layout.addWidget(self.run_button, 1,1)

        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.clicked.connect(self._stop_button_clicked)
        self.layout.addWidget(self.stop_button, 2,1)
        
        self.close_button = QtWidgets.QPushButton('Close')
        self.close_button.clicked.connect(self._close_button_clicked)
        self.layout.addWidget(self.close_button, 3,1)
        
        self.setLayout(self.layout)
    

    def get_sim_params(self):
        """ This function should be connected to a function of the parent
            object, which returns a UWBsim.SimulationParams object
        """

        print("WARNING: get_sim_params() not redefined for SimulationWindow")
        print("         > simulation using default parameters")
        print("         To use custom parameters, redefine get_sim_params() to \
            return your parameter structure")
        return self.default_params


    def query_stop_flag(self):
        """Return the state of the simulation's external stop flag"""
        return self.sim_stop_flag
    
    def run_button_clicked(self):
        """Creates a SimulationWorker to run simulation on separate Thread"""

        self.sim_stop_flag = False
        self.error_wdg.reset_errors()
        for _, plot in self.plot_list.items():
            plot.reset()

        params = self.get_sim_params()

        self.sim_worker = SimulationWorker(params, self.query_stop_flag, self._update_data)
        self.sim_worker.signals.dataUpdate.connect(self._update_data)
        self.sim_worker.signals.finished.connect(self._update_all_plots)
        self.threadpool.start(self.sim_worker)
    
    def _stop_button_clicked(self):
        """Stop simulation by setting the external stopflag """
        self.sim_stop_flag = True
    
    def _close_button_clicked(self):
        """Stop the simulation and hide the window"""
        self.sim_stop_flag = True
        self.hide()

    def _update_data(self, drone_state_data):
        """Update live error calculation and plots
        
        Callback function that takes a dictionary with the drone's time
        and state estimates and calls the individual update functions 
        for error and plots. Only redraws active plot.

        Args:
            drone_state_data: Dictionary with drone data. Must contain 
            fields 'time' and 'true'
        """

        time = drone_state_data['time'] #[-1]

        self.error_wdg.update_time(time)
        self.error_wdg.update_errors(drone_state_data)

        for _, plot in self.plot_list.items():
            plot.update_data(**drone_state_data)
        
        # Only redraw active plot for speed
        self.plotTabs.currentWidget().update_plot()
    

    def _update_all_plots(self):
        """Redraw all plots"""
        for _, plot in self.plot_list.items():
            plot.update_plot()

class QLiveInfo(QtWidgets.QWidget):
    """QtWidget that shows current simulation time and estimation errors"""

    def __init__(self, *args, **kwargs):
        super(QLiveInfo, self).__init__(*args,**kwargs)
        self.sim_time = 0.0
        self.show_errors = False
        self.error_sum2 = {}
        self.error_count = {}
        self.error_labels = {}

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        
        self.title_lb = QtWidgets.QLabel('\nSimulation Info')
        self.title_lb.setStyleSheet(" font-weight: bold ")
        self.time_lb = QtWidgets.QLabel('time \t{:.1f}'.format(self.sim_time))
        self.error_title_lb = QtWidgets.QLabel('\nRMS Errors')
        self.error_title_lb.setStyleSheet(" font-weight: bold ")
        
        self.layout.addWidget(self.title_lb)
        self.layout.addWidget(self.time_lb)
        
        self.setLayout(self.layout)


    def update_time(self, time):
        """Update the time label with the specified time"""
    
        self.sim_time = time
        self.time_lb.setText('time \t{:.1f}'.format(self.sim_time))


    def reset_errors(self):
        """Close the error labels and reset error calculation"""
        for _,label in self.error_labels.items():
            label.close()
        self.error_title_lb.setParent(None)
        self.show_errors = False
        self.error_sum2 = {}
        self.error_count = {}
        self.error_labels = {}


    def update_errors(self,estimates):
        """Update the error labels
        
        Provided with a dictionary of estimates (including an entry 'true'
        for groundtruth), calculates/updates the RMSE of all estimates 
        for position and velocity and displays them.

        Args:
            estimates: dictionary with state estimates and groundtruth (['true']) 
        """

        if not self.show_errors:
            self.show_errors = True
            self.layout.addWidget(self.error_title_lb)

        if 'true' not in estimates:
            print('Could not calculate errors: Groundtruth missing')
        else:
            for name, est in estimates.items():
                if name == 'true' or name == 'time':
                    continue
                else:
                    if not name in self.error_labels:
                        # Set up error calculation and label
                        self.error_count[name] = 0
                        self.error_sum2[name] = [0,0,0,0,0,0]
                        self.error_labels[name] = QtWidgets.QLabel('{}\n x\t{}\n y\t{}\n z\t{}\n All\t{}'.format(name, 0, 0, 0, 0))
                        self.layout.addWidget(self.error_labels[name])


                    self.error_count[name] += 1
                    self.error_sum2[name][0] += (est[0]-estimates['true'][0])**2
                    self.error_sum2[name][1] += (est[1]-estimates['true'][1])**2
                    self.error_sum2[name][2] += (est[2]-estimates['true'][2])**2
                    self.error_sum2[name][3] += (est[3]-estimates['true'][3])**2
                    self.error_sum2[name][4] += (est[4]-estimates['true'][4])**2
                    self.error_sum2[name][5] += (est[5]-estimates['true'][5])**2

                    ex = math.sqrt(self.error_sum2[name][0]/self.error_count[name])
                    ey = math.sqrt(self.error_sum2[name][1]/self.error_count[name])
                    ez = math.sqrt(self.error_sum2[name][2]/self.error_count[name])
                    exy = math.sqrt(ex**2 + ey**2)
                    e_all = math.sqrt(ex**2 + ey**2 + ez**2)
                    evx = math.sqrt(self.error_sum2[name][3]/self.error_count[name])
                    evy = math.sqrt(self.error_sum2[name][4]/self.error_count[name])
                    evz = math.sqrt(self.error_sum2[name][5]/self.error_count[name])
                    evxy = math.sqrt(evx**2 + evy**2)
                    ev_all = math.sqrt(evx**2 + evy**2 + evz**2)

                    self.error_labels[name].clear()
                    #self.error_labels[name].setText('{}\n x\t{:.3f}\n y\t{:.3f}\n z\t{:.3f}\n All\t{:.3f}'.format(name, *rmse_pos, rmse_pos_all))
                    self.error_labels[name].setText('{}\n xy\t{:.3f}\n z\t{:.3f}\n pos\t{:.3f}\n vxy\t{:.3f}\n vz\t{:.3f}\n vel\t {:.3f}'.format(name, exy, ez, e_all, evxy, evz, ev_all))


class WorkerSignals(QtCore.QObject):
    """Signals to communicate between GUI and simulation thread """
    dataUpdate = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal()

class SimulationWorker(QtCore.QRunnable):
    """Class that runs simulation on separate thread

    To avoid lags on the GUI thread, the plots are updated by the
    simulation thread at the moment (data_callback_on_this_thread()).
    Alternatively, the GUI thread can handle the callback using a signal
    (data_callback_through_signal()). In any case, to speed up the
    simulation, the full data update is not performed at every timestep.
    Instead the data is saved in an intermediate buffer.
    """

    def __init__(self, params, query_stop_flag, update_data_cb, *args, **kwargs):
        """Initializes the SimulationWorker

        Args:
            params: simulation parameters
            query_stop_flag: function that returns the state of an external stop flag
            update_data_cb: data update function that will be called from the thread
        """
        
        super(SimulationWorker, self).__init__()
        self.params = params
        self.query_stop_flag = query_stop_flag
        self.update_data_cb = update_data_cb
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.last_data_sent = 0
        self.data_buffer = {}
        self.reset_data_buffer()

    @QtCore.pyqtSlot()
    def run(self):
        """Start the simulation on separate thread
        
        Emits finished signal when simulation is done
        """
        ptvsd.debug_this_thread() 
        sim_instance = UWBSimulation(self.params, self.query_stop_flag, 
            self.data_callback_on_this_thread, **self.kwargs)

        # Emit callback signal once for GUI to setup plots
        self.data_callback_through_signal(sim_instance.drone1)
        time.sleep(1)   #Give GUI time to setup plots
        
        # Run simulation
        sim_instance.start_sim()
        self.signals.finished.emit()

    def reset_data_buffer(self):
        """Resets the data buffer"""
        self.data_buffer = {'time': [],
                            'true': []}

    def data_callback_through_signal(self, drone:Drone):
        """Emit signal to have GUI thread perform the data_callback"""
        self.data_callback(drone, self.signals.dataUpdate.emit)

    def data_callback_on_this_thread(self, drone:Drone):
        """Execute connected data callback on this thread"""
        self.data_callback(drone, self.update_data_cb)

    def data_callback(self, drone:Drone, update_function):
        """Update data_buffer and evoke data update

        This function manages the data buffer, and evokes the data update
        function at 10Hz. The update function can either be a signal 
        emission for handling on different thread, or the update function
        as a callback.

        Args:
            drone: the simulated drone object
            update_function: the function used for updating plots/errors
            with data. Can be a callback or a signal to emit
        """

        self.data_buffer['time'].append(drone.time)
        self.data_buffer['true'].append([*drone.state_true.x, *drone.state_true.v])

        for key, state in drone.state_estimate.items():
            if not key in self.data_buffer.keys():
                self.data_buffer[key] = []
            
            self.data_buffer[key].append([*state.x, *state.v])

        if (time.time() - self.last_data_sent) > 0.1:
            # for GUI, just use mean values over update interval for speed
            mean_buffer = {}
            for key, state in self.data_buffer.items():
                mean_buffer[key] = np.mean(state, axis=0)
            update_function(mean_buffer)
            self.reset_data_buffer()
            self.last_data_sent = time.time()
        

