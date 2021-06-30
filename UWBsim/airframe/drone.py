"""Drone class for simulation

This file contains the Drone class as well as the parameter structure
that is used to initialize it. In its current implementation, the drone
can only be simulated with a logfile.

Classes:
    DroneParams: Parameter structure for the configuration of the drone
    Drone: Class to represent a quadrotor drone 
"""

from typing import Dict
import numpy as np
import csv
import yaml

from UWBsim.estimators import EstimatorParams
from UWBsim.utils.dataTypes import * # pylint: disable=unused-wildcard-import
from UWBsim.utils import math3d
from UWBsim.utils.uwb_ranging import UWBGenerator, RangingSource, RangingType, RangingParams
from UWBsim.estimators.mhe import MHE
from UWBsim.estimators.ekf import EKF


class DroneParams(yaml.YAMLObject):
    """Parameter Structure for the physical aspects of the drone

    Structure for passing and saving physical parameters of the drone,
    specifically mass, drag coefficients, frame size, initial position
    and presence of altitude measurements. Inherits from YAMLObject to 
    allow saving and loading parameters from yaml files.
    
    Attributes:
        yaml_tag: tag for yaml representation
    """

    yaml_tag = u'!DroneParams'
    def __init__(self, mass=0.03, K_aero=[0.15,0.15,0.0], frame_diag=0.15,
                    frame_h=0.02, initial_pos=[0.0,0.0,0.0], 
                    altitude_enable=True, logfile=None):
        """ Initializes DroneParams

        Args:
            mass: Mass of the drone in kg
            K_aero: Drag coefficients in x,y,z (list)
            frame_diag: Frame diagonal in m
            frame_h: Frame height in m
            initial_pos: initial position of the drone in x,y,z (list, in m)
            altitude_enable: boolean, allows altitude measurements to be used
                    by the estimators
        """

        self.mass = mass
        self.K_aero = K_aero
        self.frame_diag = frame_diag
        self.frame_h = frame_h
        self.initial_pos = initial_pos
        self.altitude_enable = altitude_enable
        self.logfile = logfile

    def __repr__(self):
        return "%s(mass=%r, K_aero=%r, frame_diag=%r, frame_h=%r, \
            initial_pos=%r, altitude_enable=%r, logfile=%r)" % (
                self.__class__.__name__, self.mass, self.K_aero, 
                self.frame_diag, self.frame_h, self.initial_pos, 
                self.altitude_enable, self.logfile
            )


class Drone:
    """Class to represent a quadrotor drone in simulation
    
    The Drone class can currently only be configured to run with a 
    logfile, but it is possible to extend the class to run as a fully
    simulated drone by implementing the _step_groundtruth_sim() and 
    _get_measurements_sim() methods.

    Methods:
        step(simulation time): Executes one simulation step on the drone 
    
    """
    def __init__(self, p_drone: DroneParams, p_estimators: EstimatorParams, 
                    p_ranging:RangingParams):
        """ Initialize parameters, state variables and estimators 
        
        Args:
            p_drone: Physical parameters and configuration of the drone
            p_estimators: Parameters and configuration of the estimators
            p_ranging: Parameters and configuration of the UWB ranging
        """

        self.time = 0.0
        self.next_step = 0.0
        
        # Variables to determine if the drone has landed
        self.running = True
        self.stop_flight_counter = 0

        # Physical Constants
        self.mass = p_drone.mass
        self.frame_size = p_drone.frame_diag
        self.height = p_drone.frame_h
        self.K_aero = p_drone.K_aero

        side = self.frame_size/np.sqrt(2)
        self.inertia = (1.0/12.0) * self.mass * np.array(
            [side**2+self.height**2, side**2+self.height**2, 2*side**2])
        
        # State Variables
        self.state_true = State_XVQW()
        self.state_estimate = {}

        # Estimators
        self._setup_estimators(p_estimators)

        # Measurements
        self.altitude_enable = p_drone.altitude_enable
        self.uwb_gen = UWBGenerator(p_ranging)
        self.uwb_mode = p_ranging.rtype
        self.uwb_source = p_ranging.source
        self.anchor_enable = p_ranging.anchor_enable
        self.anchor_pos = p_ranging.anchor_positions
        if self.anchor_enable is not None:
            self.N_anchors = len(self.anchor_enable)
        else:
            self.N_anchors = 0
        self.twr_history = [[] for _ in range(self.N_anchors)]
        self.tdoa_history = [[[] for _ in range(self.N_anchors)] for _ in range(self.N_anchors)]
        self.alt_history = []

        self.logfile = p_drone.logfile
        if self.logfile is None:
            # TODO: implement fully simulated drone
            self.use_log = False
            print("Fully simulated drone not implemented yet")
            raise ValueError
        else:
            self.use_log = True
            self.log = open(self.logfile, 'r', newline='')
            self.logreader = csv.DictReader(self.log, skipinitialspace=True)

            if self.uwb_mode == RangingType.TWR:
                self.last_uwb = np.zeros(self.N_anchors)

            elif self.uwb_mode == RangingType.TDOA:
                self.last_uwb = np.zeros((self.N_anchors, self.N_anchors))

            # Fast forward until optitrack fix
            logline = next(self.logreader)
            while ( float(logline['otX'])==0 and float(logline['otY'])==0 and
                    float(logline['otZ'])==0 ):
                logline = next(self.logreader)

            self.log_data = self._logline_to_data(logline)
            self.next_data = self._logline_to_data(next(self.logreader))
            self.t0 = self.log_data['timeTick']


    def __exit__(self, type, value, traceback):
        self._stop()
    

    def _stop(self):
        """ Close log file to safely exit the program """
        if self.use_log:
            self.log.close()


    def _logline_to_data(self, logline: Dict):
        """Change data type of log variables from string to float

        Args:
            logline: dictionary where field names are header entries and
            values are variables

        Returns:
            Dictionary with same keys but with values changed to float
            where possible 
        
        Raises:
            TypeError: logline is not complete and contains None types
        """

        data = {}
        for key, value in logline.items():
            data[key] = float(value)

        return data


    def _setup_estimators(self, params: EstimatorParams):
        """Setup estimators and variables for state estimates

        Estimators are created and tuned according to specifications in 
        params. All objects (estimators, state estimates, dt, tnext) are
        saved in dictionaries. 

        Args:
            params: configuration and parameters of the estimators
        """

        self.estimator_isEnabled = {'mhe': False,
                                    'ekf': False
                                    }
        self.estimators = {}
        self.est_dt = {}
        self.est_tnext = {}
     
        if params.mhe.enable:
            rate_mhe = params.mhe.rate
            self.estimator_isEnabled['mhe'] = True
            self.est_dt['mhe'] = 1.0/rate_mhe
            self.est_tnext['mhe'] = self.est_dt['mhe']
            self.estimators['mhe'] = MHE(params.mhe, self.mass, self.K_aero)
            self.state_estimate['mhe'] = State_XVQW()

        if params.ekf.enable:
            rate_ekf = params.ekf.rate
            self.estimator_isEnabled['ekf'] = True
            self.est_dt['ekf'] = 1.0/rate_ekf
            self.est_tnext['ekf'] = self.est_dt['ekf']
            self.estimators['ekf'] = EKF(params.ekf, self.mass, self.K_aero)
            self.state_estimate['ekf'] = State_XVQW()


    def step(self, sim_time):
        """ Simulate the next time step.
        
        First propagates the groundtruth and then gather measurements
        and simulate the estimators.

        Args:
            sim_time: current simulation time
        
        Returns:
            True if drone is still flying, False if end of simulated
            path/log is reached.
        """
        
        if sim_time >= self.next_step:
            self.time = sim_time
            if self.use_log:
                if (self.next_data['timeTick']-self.t0)/1000 <= sim_time:
                    self.log_data = self.next_data
                    try:
                        self.next_data = self._logline_to_data(next(self.logreader))
                    except (StopIteration, TypeError, ValueError):
                        self.running = False

                self._step_groundtruth_log(self.log_data)
                m = self._get_measurements_log(self.log_data)
                self._step_estimators(m)
                self.next_step = sim_time + 0.01

            else:
                self._step_groundtruth_sim()
                m = self._get_measurements_sim()
                self._step_estimators(m)
                self.next_step = sim_time + 0.01

        if not self.running:
            self._stop()

        return self.running


    def _step_groundtruth_sim(self):
        """NOT IMPLEMENTED

        Propagate groundtruth through full simulation
        """
        return NotImplemented
    

    def _get_measurements_sim(self):
        """NOT IMPLEMENTED 
        
        Generate measurements from fully simulated Drone 
        """
        return NotImplemented


    def _step_groundtruth_log(self, log_data):
        """Propagate groundtruth from log 
        
        Update the drone's true state using the log_data provided

        Args:
            log_data: Dictionary with groundtruth information.
        """

        dt = self.time - self.state_true.timestamp
        
        # Get current position
        tmpx = np.array([log_data['otX'], log_data['otY'], log_data['otZ']])
        if any([tmpx[0] is None, tmpx[1] is None, tmpx[2] is None]):
            tmpx = np.array([0.0,0.0,0.0])
        
        # stop drone when landed
        if (self.time>10 and tmpx[2]<0.1 and 
                np.linalg.norm(self.state_true.x - tmpx)<0.001):
            self.stop_flight_counter += 1
            if self.stop_flight_counter > 200:
                self.running = False
        else:
            self.stop_flight_counter = 0

        # Calculate (angular) velocity
        if dt==0:
            self.state_true.v[0] = 0
            self.state_true.v[1] = 0
            self.state_true.v[2] = 0
        else:
            self.state_true.v[0] = (tmpx[0]-self.state_true.x[0])/dt
            self.state_true.v[1] = (tmpx[1]-self.state_true.x[1])/dt
            self.state_true.v[2] = (tmpx[2]-self.state_true.x[2])/dt
        
        # update true state position and quaternion
        self.state_true.x[0] = tmpx[0]
        self.state_true.x[1] = tmpx[1]
        self.state_true.x[2] = tmpx[2]

        self.state_true.timestamp = self.time


    def _get_measurements_log(self, log_data):
        """Extract measurements from provided log_data
        
        Reads gyro and acc data from log_data and turns them into an 
        IMU_meas object. If altitude measurements are enabled, also
        generates altitude measurements. UWB measurements are created
        based on ranging parameters (twr/tdoa, from log/generated).

        Args:
            log_data: Dictionary containing measurements
        
        Returns:
            Array of different types of measurements
        """

        measurements = []

        # IMU
        gyro = [0,0,0]
        gyro[0] = log_data['gyroX'] * np.pi / 180.0
        gyro[1] = log_data['gyroY'] * np.pi / 180.0
        gyro[2] = log_data['gyroZ'] * np.pi / 180.0

        acc = [0,0,0]
        acc[0] = log_data['accX'] * 9.813
        acc[1] = log_data['accY'] * 9.813
        acc[2] = log_data['accZ'] * 9.813
        
        measurements.append(IMU_meas(gyro, acc, self.time))
        
        # UWB
        if self.uwb_mode == RangingType.TWR:
            for anchor_id in range(self.N_anchors):
                twr = self._get_twr(anchor_id, log_data)
                if twr is not None:
                    measurements.append(twr)
        elif self.uwb_mode == RangingType.TDOA:
            for anchor_idA in range(self.N_anchors):
                for anchor_idB in range(anchor_idA):
                    tdoa = self._get_tdoa(anchor_idA, anchor_idB, log_data)
                    if tdoa is not None:
                        measurements.append(tdoa)
        
        # Altitude
        if self.altitude_enable:
            alt = log_data['otZ'] * (1 + np.random.normal(0.0, 0.02))
            alt_measurement = Alt_meas(alt, stdDev=0.02, timestamp=self.time)
            alt_measurement.stdDev = self._get_meas_stdev_real(alt_measurement)
            measurements.append(alt_measurement)

        return measurements

        
    def _step_estimators(self, measurements):
        """Simulate all enabled estimators

        Provides inputs and measurements to the estimators and calls
        their individual step functions according to their estimation
        frequency.

        Args:
            measurements: Array of various types of measurements
        """

        # Add measurements/inputs
        for m in measurements:
            for _,estimator in self.estimators.items():
                if isinstance(m, IMU_meas):
                    thrust = m.acc[2]*self.mass
                    estimator.addInputs(thrust, m.gyro, m.acc, m.timestamp)
                else:
                    estimator.addMeasurement(m)
                
        # Run estimators
        for key,estimator in self.estimators.items():
            if self.time >= self.est_tnext[key]:
                self.est_tnext[key] += self.est_dt[key]
                self.state_estimate[key] = estimator.step(self.time)


    def _get_twr(self, anchor_id, log_data=None):
        """Create TWR measurement for a specific anchor

        Creates a TWR measurement according to the ranging parameters
        set in the drone. Returns None if no new measurement is available
        for the anchor.

        Args:
            anchor_id: ID of the anchor for which to create the measurement
            log_data: Dictionary containing twr measurements

        Returns:
            TWR_meas if new measurement is available for the given anchor,
            None otherwise.
        """

        twr = None
        if self.anchor_enable[anchor_id]:
            if self.uwb_source == RangingSource.LOG:
                key = 'twr{}'.format(anchor_id)
                if key in log_data: 
                    dist = log_data[key]
                    if dist != self.last_uwb[anchor_id]:
                        self.last_uwb[anchor_id] = dist
                        twr = TWR_meas(self.anchor_pos[anchor_id], 
                                anchor_id, dist, timestamp=self.time)
            else: # Ranging source not Log
                twr = self.uwb_gen.generate_twr(self.state_true.x, anchor_id, self.time)

        if twr is not None:    
            twr.stdDev = self._get_meas_stdev_real(twr)
        return twr


    def _get_tdoa(self, anchor_idA, anchor_idB, log_data=None):
        """Create TDOA measurement for a specific anchor

        Creates a TDOA measurement according to the ranging parameters
        set in the drone. Returns None if no new measurement is available
        for the anchor.

        Args:
            anchor_idA: ID of anchorA for which to create the measurement
            anchor_idB: ID of anchorB for which to create the measurement
            log_data: Dictionary containing tdoa measurements

        Returns:
            TWR_meas if new measurement is available for the given anchor,
            None otherwise.
        """
        
        tdoa = None
        if self.anchor_enable[anchor_idA] and self.anchor_enable[anchor_idB]:
            if self.uwb_source == RangingSource.LOG:
                key = 'tdoa{}{}'.format(anchor_idB, anchor_idA)
                if key in log_data:
                    distDiff = log_data[key]
                    if distDiff != self.last_uwb[anchor_idA][anchor_idB]:
                        self.last_uwb[anchor_idA][anchor_idB] = distDiff
                        tdoa = TDOA_meas(self.anchor_pos[anchor_idA], 
                            self.anchor_pos[anchor_idB], anchor_idA, anchor_idB,
                            -distDiff, timestamp=self.time)
            else: # Ranging source not Log
                tdoa = self.uwb_gen.generate_tdoa(self.state_true.x, anchor_idA,
                        anchor_idB, self.time)
        if tdoa is not None:
            tdoa.stdDev = self._get_meas_stdev_real(tdoa)
        return tdoa
        

    def _get_meas_stdev(self, measurement):
        """Calculate Standard Deviation of measurements (approximate)

        Adds a measurement to a history of recent measurements from the
        same sensor and then calculates the standard deviation of the 
        measurements in the history. Assumes that all measurements in the
        history are taken at the same location.

        Args:
            measurement: Measurement of the sensor for which to compute
            the standard deviation
        
        Returns:
            Standard Deviation of recent measurements from the sensor
        """

        stdDev = measurement.stdDev
        if isinstance(measurement, Alt_meas):
            self.alt_history.append(measurement.alt)
            if len(self.alt_history) > 5:
                stdDev = np.std(self.alt_history)
            if len(self.alt_history) > 10:
                self.alt_history.pop(0)
        elif isinstance(measurement, TWR_meas):
            i = measurement.anchor_id
            self.twr_history[i].append(measurement.distance)
            if len(self.twr_history[i]) > 5:
                stdDev = np.std(self.twr_history[i])
            else:
                stdDev = 0.2
            if len(self.twr_history[i]) > 10:
                self.twr_history[i].pop(0)
        elif isinstance(measurement, TDOA_meas):
            i = measurement.anchorA_id
            j = measurement.anchorB_id
            self.tdoa_history[i][j].append(measurement.distDiff)
            if len(self.tdoa_history[i][j]) > 5:
                stdDev = np.std(self.tdoa_history[i][j])
            else:
                stdDev = 1.0
            if len(self.tdoa_history[i][j]) > 10:
                self.tdoa_history[i][j].pop(0)
        return stdDev


    def _get_meas_stdev_real(self, measurement):
        """Calculate Standard Deviation of measurements (True)

        Adds error of the measurement to a history of recent measurement
        errors from the same sensor and then calculates the standard 
        deviation of the measurement errors in the history. Requires
        knowledge of groundtruth and is therefore less realistic

        Args:
            measurement: Measurement of the sensor for which to compute
            the standard deviation
        
        Returns:
            Standard Deviation of recent measurements from the sensor
        """

        stdDev = measurement.stdDev
        if isinstance(measurement, Alt_meas):
            alt_error = measurement.alt - self.state_true.x[2]
            self.alt_history.append(alt_error)
            if len(self.alt_history) > 5:
                stdDev = np.std(self.alt_history)
            else:
                stdDev = 0.02
            if len(self.alt_history) > 10:
                self.alt_history.pop(0)
        elif isinstance(measurement, TWR_meas):
            i = measurement.anchor_id
            twr_true = np.linalg.norm(measurement.anchor-self.state_true.x)
            twr_error = measurement.distance - twr_true
            self.twr_history[i].append(twr_error)
            if len(self.twr_history[i]) > 5:
                stdDev = np.std(self.twr_history[i])
            else:
                stdDev = 0.2
            if len(self.twr_history[i]) > 10:
                self.twr_history[i].pop(0)
        elif isinstance(measurement, TDOA_meas):
            i = measurement.anchorA_id
            j = measurement.anchorB_id
            tdoa_true = np.linalg.norm(measurement.anchorB-self.state_true.x) \
                        - np.linalg.norm(measurement.anchorA-self.state_true.x)
            tdoa_error = measurement.distDiff - tdoa_true
            self.tdoa_history[i][j].append(tdoa_error)
            if len(self.tdoa_history[i][j]) > 5:
                stdDev = np.std(self.tdoa_history[i][j])
            else:
                stdDev = 1.0
            if len(self.tdoa_history[i][j]) > 10:
                self.tdoa_history[i][j].pop(0)
        return stdDev
