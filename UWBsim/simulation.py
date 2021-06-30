"""Top level simulation functionality to be used in external UI

This file contains the top level simulation functionality that can be
used to create and run (UWB)-Simulations from any external interface.

Classes:
    SimulationParams: Data class to transfer and save simulation parameters
    UWBSimulation: The class performing the simulation
"""

import yaml
import numpy as np

from UWBsim.estimators import EstimatorParams
from UWBsim.airframe.drone import Drone, DroneParams
from UWBsim.utils.uwb_ranging import RangingParams

class SimulationParams(yaml.YAMLObject):
    """Parameter Structure for the simulation

    Structure for passing and saving parameters used by the simulation,
    specifically logfile, name, drone parameters, estimator parameters
    and ranging parameters. Inherits from YAMLObject to allow saving 
    and loading parameters from yaml files.
    
    Attributes:
        yaml_tag: tag for yaml representation
    """

    yaml_tag = u'!SimulationParams'
    def __init__(self, name='', drone=DroneParams(), estimators=EstimatorParams(),
                    ranging=RangingParams()):
        """ Initializes SimulationParams

        Args:
            name: simulation runs can be named to simplify identification of runs
            drone: DroneParams() object with parameters for the drone
            estimators: EstimatorParams() object with parameters for the estimator(s) used
            ranging: RangingParams() object with parameters for UWB ranging
        """

        self.name = name
        self.drone = drone
        self.ranging = ranging
        self.estimators = estimators

    def __repr__(self):
        return "%s(name=%r, drone=%r, estimators=%r, ranging=%r)" % (
                self.__class__.__name__, self.name, self.drone, 
                self.estimators, self.ranging 
            )

class UWBSimulation:
    """Class to handle simulation of state estimators using UWB

    This class takes care of setting up and executing the simulation of a drone
    with estimators. The simulation is set up based on the parameters provided to the
    initialization function. Optionally, external functions can be provided to
    terminate the simulation early (query_stop_flag) and to evaluate data in real-time
    (data_callback). Both query_stop_flag() and data_callback() are called once per time step.
    
    Methods:
        start_sim(): Starts running the simulation
    """

    def __init__(self, params: SimulationParams, query_stop_flag=NotImplemented, 
                    data_callback=NotImplemented, time_step = 0.001):
        """Initializes the UWBSimulation

        Args:
            params: Instance of SimulationParams, containing all relevant 
                    simulation parameters
            query_stop_flag: Function which returns the status of an external 
                    stop flag. Simulation will stop when this function returns 
                    'True' (optional)
            data_callback: Callback function that processes live drone data (optional)
            time_step: Simulation step size in seconds
        
        Raises:
            ValueError: No logfile specified
        """

        self.time_step = time_step
        self.params = params

        if params.drone.logfile is None:
            print("Missing Logfile - Simulation without Logfile not implemented")
            raise ValueError
        else:
            self.drone1 = Drone(params.drone, params.estimators, params.ranging)
        
        if query_stop_flag is NotImplemented:
            self.stop_flag = lambda: False
        else:
            self.stop_flag = query_stop_flag

        if data_callback is NotImplemented:
            self.data_callback = lambda: None
        else:
            self.data_callback = data_callback


    def start_sim(self):
        """Starts the simulation

            Simulates the simulation objects at regular time intervals (specified
            during initialization of UWBSimulation). Simulation objects are simulated
            by calling their step() functions with the current simulation time. The 
            simulation terminates when the drone's step() function returns 'False'.
        """

        step_counter = 0
        while True:
            sim_time = step_counter * self.time_step
            step_counter += 1

            if (10*sim_time)%1==0:
                # Only update console every 0.1s
                print("{}: [{:2.1f}s]".format(self.params.name, sim_time), end='\r')

            if not self.drone1.step(sim_time):
                print("{}: DONE             ".format(self.params.name))
                break

            self.data_callback(self.drone1)
            if self.stop_flag():
                print("{}: CANCELLED         ".format(self.params.name))
                break
        