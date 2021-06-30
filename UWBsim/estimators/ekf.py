"""Extended Kalman Filter

This file contains the EKF for state estimation as well as the 
datastructure for passing parameters to the EKF. The EKF implementation
follows closely the implementation on the Crazyflie Drone 
(https://github.com/bitcraze/crazyflie-firmware). The corresponding
publication is by Mueller, Hamer and D'Andrea: 
    Mueller, M. W., Hamer, M., & D’Andrea, R. (2015). Fusing ultra-wideband 
    range measurements with accelerometers and rate gyroscopes for 
    quadrocopter state estimation. 2015 IEEE International Conference on 
    Robotics and Automation (ICRA), 1730–1736. 
    https://doi.org/10.1109/ICRA.2015.7139421

Classes:
    EKF_Params: The parameter data structure
    EKF: The Moving Horizon Estimator
"""
import yaml
import numpy as np
import time as timing

from UWBsim.utils import math3d
from UWBsim.utils.dataTypes import State_XVQW, TWR_meas, TDOA_meas, Alt_meas

STATE_X = 0
STATE_Y = 1
STATE_Z = 2
STATE_PX = 3
STATE_PY = 4
STATE_PZ = 5
STATE_D0 = 6
STATE_D1 = 7
STATE_D2 = 8 

MAX_COV = 100
MIN_COV = 0.000001

G = 9.81

stdDev_initialPos_xy = 100
stdDev_initialPos_z = 1
stdDev_initialVelocity = 0.01
stdDev_initialAtt_rp = 0.01
stdDev_initialAtt_y = 0.01

procNoise_Acc_xy = 0.5
procNoise_Acc_z = 1.0
procNoise_Vel = 0
procNoise_Pos = 0
procNoise_Att = 0
measNoise_Gyro = 0.1

class EKF_Params(yaml.YAMLObject):
    """Parameter Structure for the EKF

    Structure for passing and saving initialization parameters 
    for the EKF. Inherits from YAMLObject to allow saving and loading 
    parameters from yaml files.
    
    Attributes:
        yaml_tag: tag for yaml representation
    """

    yaml_tag = u'!EKF_Params'

    def __init__(self, enable=False, rate=100, outlierThreshold = 12):
        """ Initializes EKF_Params

            Default initialization parameters are the same as on the CF.

            Args:
                enable: boolean, indicates if the EKF shall be enabled
                rate: update rate of the EKF
        """
        self.enable = enable
        self.rate = rate
        self.outlierThreshold = outlierThreshold
    
    def __repr__(self):
        return "%s(enable=%r, rate=%r, outlierThreshold=%r)" % (
            self.__class__.__name__, self.enable, self.rate, 
            self.outlierThreshold
        )

class EKF:
    """EKF for state estimation on a drone (Position)

    The EKF implementation follows closely the implementation on the
    Crazyflie Drone (https://github.com/bitcraze/crazyflie-firmware. 
    
    Usage: after initialization, use addInputs() and addMeasurements() 
    to add input/measurement data to the buffer. Call step() to update 
    the current estimate at t.

    Methods:
        reset(): reset the estimator
        step(): Execute one estimation step and return current state
        getEstimateNow(): Get the current state after the last update
        addInputs(): Add inputs to the EKF input accumulators
        addMeasurement(): Add a measurement to the measurement queues
    """

    def __init__(self, ekf_params: EKF_Params, mass, K_aero, 
                    initial_state=State_XVQW()):
        """Initializes the EKF

        Args:
            ekf_params: Instance of EKF_Params containing 
            the estimator parameters
            mass: Mass of the drone
            K_aero: 3x3 drag matrix
            initial_state: state at which the estimator is initialized
        """

        self.xi = np.zeros((9,1))
        self.q = np.zeros(4)
        self.R = np.eye(3)
        self.gyro = np.zeros(3)

        self.m = mass
        self.K_aero = K_aero

        self.reset(initial_state)
        self.outlierThreshold = ekf_params.outlierThreshold
        self.method_time = []

    def reset(self, initial_state):
        """Resets the estimator to the initial state

        Resets all internal variables to the same values as on startup 
        and sets the provided initial state as state.

        Args:
            initial_state: State_XVQW state to which to reset
        """

        self.t = 0
        self.isFlying = False
        
        # set initial/reset state
        self.xi[STATE_X] = initial_state.x[0]
        self.xi[STATE_Y] = initial_state.x[1]
        self.xi[STATE_Z] = initial_state.x[2]
        
        self.q[0] = initial_state.q[0]
        self.q[3] = initial_state.q[3]

        # reset state variances
        self.P = np.zeros((9,9))
        self.P[STATE_X][STATE_X] = stdDev_initialPos_xy**2
        self.P[STATE_Y][STATE_Y] = stdDev_initialPos_xy**2
        self.P[STATE_Z][STATE_Z] = stdDev_initialPos_z**2
        self.P[STATE_PX][STATE_PX] = stdDev_initialVelocity**2
        self.P[STATE_PY][STATE_PY] = stdDev_initialVelocity**2
        self.P[STATE_PZ][STATE_PZ] = stdDev_initialVelocity**2
        self.P[STATE_D0][STATE_D0] = stdDev_initialAtt_rp**2
        self.P[STATE_D1][STATE_D1] = stdDev_initialAtt_rp**2
        self.P[STATE_D2][STATE_D2] = stdDev_initialAtt_y**2
        
        # reset measurement queues
        self.twr_queue = []
        self.tdoa_queue = []
        self.alt_queue = []
        self.tdoa_count = 0

        self.inputCounter = 0
        self.inputAccum_f = 0
        self.inputAccum_w = np.zeros(3)
    

    def step(self, sim_time):
        """ Execute one estimation step of the EKF and return current estimate

        Uses the accumulated inputs to perform an EKF prediction step,
        then corrects the state estimate with all queued measurements.

        Args:
            time: current time of the simulation
        
        Returns:
            Current state estimate at 'time' in XVQW_State format
        
        Raises:
            AssertionError: State estimate is out of bounds or NaN
        """

        tic = timing.perf_counter_ns()
        dt = sim_time - self.t
        self.t = sim_time

        # Get inputs from accumulators
        self.inputCounter = self.inputCounter if self.inputCounter != 0 else 1
        avg_f = self.inputAccum_f / self.inputCounter
        avg_w = self.inputAccum_w / self.inputCounter
        acc = np.zeros(3)

        # predict state and reset accumulators
        self._predict(avg_f, acc, avg_w, dt)
        self.inputCounter = 0
        self.inputAccum_f = 0
        self.inputAccum_w = np.zeros(3)
        
        if dt>0:
            self._addProcessNoise(dt)
        
        # update state with measurements from queues
        while len(self.twr_queue)>0:
            self._updateWithTWR(self.twr_queue[0])
            self.twr_queue = self.twr_queue[1:]

        while len(self.tdoa_queue)>0:
            self._updateWithTDOA(self.tdoa_queue[0])
            self.tdoa_queue = self.tdoa_queue[1:]
        
        while len(self.alt_queue)>0:
            self._updateWithAlt(self.alt_queue[0])
            self.alt_queue = self.alt_queue[1:]
        
        # finalize state and check validity
        self._finalize()
        EKF_ASSERT_STATE_VALID(self.xi, self.P)
        
        est = self.get_estimate_now(sim_time)
        toc = timing.perf_counter_ns()
        self.method_time.append(toc-tic)
        return est
    
    def addInputs(self, thrust, gyro, acc, time):
        """Add inputs to EKF input accumulators

        Adds inputs to the EKF input accumulators which act as buffer 
        between steps of the estimator. The accumulators are emptied by 
        the EKF step function, which averages the inputs to a single 
        value (per input) per timestep. 

        Args:
            thrust: thrust measurement in N, float
            gyro: 3D gyro measurements in radians, list(3)
            acc: 3D accelerometer measurement in m/s2, list(3)
            time: current time, float
        """

        if not self.isFlying and thrust>=0.99*G*self.m:
            self.isFlying = True
        
        self.inputCounter += 1
        self.inputAccum_f += thrust
        self.inputAccum_w += gyro

    
    def addMeasurement(self, measurement):
        """Add a measurement to the EKF measurement queues

        Adds any known measurement to the respective measurement queue
        of the EKF. The measurement queue is emptied by the step()
        function when updating the estimated state. 

        Args:
            measurement: any type of measurement that can be used by the
            estimator
        """

        if isinstance(measurement, TWR_meas):
            self.twr_queue.append(measurement)
        elif isinstance(measurement, TDOA_meas):
            self.tdoa_queue.append(measurement)
        elif isinstance(measurement, Alt_meas):
            self.alt_queue.append(measurement)
        else:
            print("[EKF]: Unknown measurement type received {}".format(
                type(measurement)
            ))


    def _predict(self, thrust, acc, gyro, dt):
        """Performs an EKF prediction step

        Linearizes the system dynamics at the current state and predicts
        the next state based on the provided inputs.

        Args:
            thrust: current thrust
            acc: current accelerometer measurement
            gyro: current gyroscope measurements
            dt: time since last prediction
        """
        
        # Dynamics linearization
        # Build A from 3x3 blocks
        A_eye = np.eye(3)
        A_zero = np.zeros((3,3))

        A_dxdp = self.R*dt
        rho_skew = math3d.skew([self.xi[STATE_PX][0], self.xi[STATE_PY][0], self.xi[STATE_PZ][0]])
        A_dxdd = -np.matmul(self.R, rho_skew) * dt

        A_dpdp = np.eye(3) - math3d.skew(gyro) * dt
        A_dpdd = -G * math3d.skew([self.R[2][0], self.R[2][1], self.R[2][2]]) * dt

        d = gyro*dt/2
        tmp = np.array([[ -d[1]*d[1]-d[2]*d[2],  d[0]*d[1],  d[0]*d[2]],
                        [  d[0]*d[1], -d[0]*d[0]-d[2]*d[2],  d[1]*d[2]],
                        [  d[0]*d[2],  d[1]*d[2], -d[0]*d[0]-d[1]*d[1]]])
        A_dddd = np.eye(3) - math3d.skew(d) + 0.5*tmp

        A = np.block([[A_eye, A_dxdp, A_dxdd],
                      [A_zero, A_dpdp, A_dpdd],
                      [A_zero, A_zero, A_dddd]])

        # Covariance Update
        self.P = np.matmul(A, np.matmul(self.P, A.T))

        if self.isFlying:
            zacc = thrust/self.m

            dx = self.xi[STATE_PX][0] * dt
            dy = self.xi[STATE_PY][0] * dt
            dz = self.xi[STATE_PZ][0] * dt + zacc * dt*dt /2.0

            self.xi[STATE_X][0] += self.R[0][0]*dx + self.R[0][1]*dy + self.R[0][2]*dz
            self.xi[STATE_Y][0] += self.R[1][0]*dx + self.R[1][1]*dy + self.R[1][2]*dz
            self.xi[STATE_Z][0] += self.R[2][0]*dx + self.R[2][1]*dy + self.R[2][2]*dz - G*dt*dt/2.0

            tmpP = np.array([self.xi[STATE_PX][0], self.xi[STATE_PY][0], self.xi[STATE_PZ][0]])
            
            self.xi[STATE_PX][0] += dt * (gyro[2]*tmpP[1] - gyro[1]*tmpP[2] - G*self.R[2][0])
            self.xi[STATE_PY][0] += dt * (gyro[0]*tmpP[2] - gyro[2]*tmpP[0] - G*self.R[2][1])
            self.xi[STATE_PZ][0] += dt * (zacc + gyro[1]*tmpP[0] - gyro[0]*tmpP[1] - G*self.R[2][2])

        
        else:
            dx = self.xi[STATE_PX][0] * dt + acc[0]*dt*dt/2.0
            dy = self.xi[STATE_PY][0] * dt + acc[1]*dt*dt/2.0
            dz = self.xi[STATE_PZ][0] * dt + acc[2]*dt*dt/2.0

            self.xi[STATE_X][0] += self.R[0][0]*dx + self.R[0][1]*dy + self.R[0][2]*dz
            self.xi[STATE_Y][0] += self.R[1][0]*dx + self.R[1][1]*dy + self.R[1][2]*dz
            self.xi[STATE_Z][0] += self.R[2][0]*dx + self.R[2][1]*dy + self.R[2][2]*dz - G*dt*dt/2.0

            tmpP = np.array([self.xi[STATE_PX][0], self.xi[STATE_PY][0], self.xi[STATE_PZ][0]])

            self.xi[STATE_PX][0] += dt * (acc[0] + gyro[2]*tmpP[1] - gyro[1]*tmpP[2] - G*self.R[2][0])
            self.xi[STATE_PY][0] += dt * (acc[1] + gyro[0]*tmpP[2] - gyro[2]*tmpP[0] - G*self.R[2][1])
            self.xi[STATE_PZ][0] += dt * (acc[2] + gyro[1]*tmpP[0] - gyro[0]*tmpP[1] - G*self.R[2][2])

        # Attitude update
        dtw = dt*gyro
        angle = np.linalg.norm(dtw)
        dq = math3d.aa2quat(angle, dtw)

        tmpq = np.zeros(4)
        tmpq[0] = dq[0]*self.q[0] - dq[1]*self.q[1] - dq[2]*self.q[2] - dq[3]*self.q[3]
        tmpq[1] = dq[1]*self.q[0] + dq[0]*self.q[1] + dq[3]*self.q[2] - dq[2]*self.q[3]
        tmpq[2] = dq[2]*self.q[0] - dq[3]*self.q[1] + dq[0]*self.q[2] + dq[1]*self.q[3]
        tmpq[3] = dq[3]*self.q[0] + dq[2]*self.q[1] - dq[1]*self.q[2] + dq[0]*self.q[3]
        
        norm = np.linalg.norm(tmpq)
        self.q = tmpq/norm


    def _addProcessNoise(self, dt):
        """Add process noise to the current covariance matrix
        
        Updates the covariance matrix after a prediction step

        Args:
            dt: time between predictions
        """

        if dt>0:
            self.P[STATE_X][STATE_X] += (procNoise_Acc_xy*dt*dt + procNoise_Vel*dt + procNoise_Pos)**2
            self.P[STATE_Y][STATE_Y] += (procNoise_Acc_xy*dt*dt + procNoise_Vel*dt + procNoise_Pos)**2
            self.P[STATE_Z][STATE_Z] += (procNoise_Acc_z*dt*dt + procNoise_Vel*dt + procNoise_Pos)**2

            self.P[STATE_PX][STATE_PX] += (procNoise_Acc_xy*dt + procNoise_Vel)**2
            self.P[STATE_PY][STATE_PY] += (procNoise_Acc_xy*dt + procNoise_Vel)**2
            self.P[STATE_PZ][STATE_PZ] += (procNoise_Acc_z*dt + procNoise_Vel)**2

            self.P[STATE_D0][STATE_D0] += (measNoise_Gyro*dt + procNoise_Att)**2
            self.P[STATE_D1][STATE_D1] += (measNoise_Gyro*dt + procNoise_Att)**2
            self.P[STATE_D2][STATE_D2] += (measNoise_Gyro*dt + procNoise_Att)**2

        for i in range(9):
            for j in range(i,9):
                p = 0.5*self.P[i][j] + 0.5*self.P[j][i]
                if np.isnan(p) or p>MAX_COV:
                    self.P[i][j] = MAX_COV
                    self.P[j][i] = MAX_COV
                elif i==j and p<MIN_COV:
                    self.P[i][j] = MIN_COV
                    self.P[j][i] = MIN_COV
                else:
                    self.P[i][j] = p
                    self.P[j][i] = p


    def _finalize(self):
        """Move attitude delta into the quaternion and rotation matrix

        Converts the altitude delta states into a delta quaternion which
        is used to update the reference attitude (quaternion and rotation
        matrix). After this, the altitude delta states are reset to 0 and
        the covariance matrix is updated.
        """

        v = np.array([self.xi[STATE_D0][0], self.xi[STATE_D1][0], self.xi[STATE_D2][0]])
        if (abs(v[0])>0.001 or abs(v[1])>0.001 or abs(v[2])>0.001) and (abs(v[0])<10 and abs(v[1])<10 and abs(v[2])<10):
            angle = np.linalg.norm(v)
            ca = np.cos(angle/2.0)
            sa = np.sin(angle/2.0)
            dq = np.array([ca, sa*v[0]/angle, sa*v[1]/angle, sa*v[2]/angle])
            
            tmpq = np.zeros(4)
            tmpq[0] = dq[0]*self.q[0] - dq[1]*self.q[1] - dq[2]*self.q[2] - dq[3]*self.q[3]
            tmpq[1] = dq[1]*self.q[0] + dq[0]*self.q[1] + dq[3]*self.q[2] - dq[2]*self.q[3]
            tmpq[2] = dq[2]*self.q[0] - dq[3]*self.q[1] + dq[0]*self.q[2] + dq[1]*self.q[3]
            tmpq[3] = dq[3]*self.q[0] + dq[2]*self.q[1] - dq[1]*self.q[2] + dq[0]*self.q[3]
        
            norm = np.linalg.norm(tmpq)
            self.q = tmpq/norm

            d = v/2

            A = np.eye(9)
            A[STATE_D0][STATE_D0] = 1 -d[1]*d[1]/2 - d[2]*d[2]/2
            A[STATE_D0][STATE_D1] =  d[2] + d[0]*d[1]/2
            A[STATE_D0][STATE_D2] = -d[1] + d[0]*d[2]/2

            A[STATE_D1][STATE_D0] = -d[2] + d[0]*d[1]/2
            A[STATE_D1][STATE_D1] = 1 -d[0]*d[0]/2 - d[2]*d[2]/2
            A[STATE_D1][STATE_D2] =  d[0] + d[1]*d[2]/2

            A[STATE_D2][STATE_D0] =  d[1] + d[0]*d[2]/2
            A[STATE_D2][STATE_D1] = -d[0] + d[1]*d[2]/2
            A[STATE_D2][STATE_D2] = 1 -d[0]*d[0]/2 - d[1]*d[1]/2

            self.P = np.matmul(A, np.matmul(self.P, A.T))

        quat = math3d.Quaternion([self.q[0],self.q[1],self.q[2],self.q[3]])
        self.R = quat.rotation_matrix()

        self.xi[STATE_D0] = 0
        self.xi[STATE_D1] = 0
        self.xi[STATE_D2] = 0

        for i in range(9):
            for j in range(i,9):
                p = 0.5*self.P[i][j] + 0.5*self.P[j][i]
                if np.isnan(p) or p>MAX_COV:
                    self.P[i][j] = MAX_COV
                    self.P[j][i] = MAX_COV
                elif i==j and p<MIN_COV:
                    self.P[i][j] = MIN_COV
                    self.P[j][i] = MIN_COV
                else:
                    self.P[i][j] = p
                    self.P[j][i] = p


    def _scalarUpdate(self, Hm, error, stdMeasNoise, filterOutliers=False):
        """Performs a measurement update with a scalar

        Updates the state estimate and covariance matrix with a single,
        scalar measurement. This makes the update quite efficient,
        because no matrix inversion is needed.

        Args:
            Hm: Jacobian of the measurement equation
            error: Measurement error (measured-predicted)
            stdMeasNoise: Standard deviation of the measurement noise
        """
        R = stdMeasNoise**2

        # Kalman Gain
        PHT = np.matmul(self.P, np.transpose(Hm))
        HPHR = R
        for i in range(9):
            HPHR += Hm[0][i]*PHT[i][0]
        
        # Mahalanobis Outlier filter
        if filterOutliers and abs(error)/HPHR > self.outlierThreshold:
            return
        

        K = np.matmul(self.P, Hm.T)/HPHR  # (PH' (HPH'+R)^-1)

        # State update
        for i in range(9):
            self.xi[i][0] = self.xi[i][0] + K[i][0]*error
        
        # Covariance update
        KHI = np.matmul(K,Hm) - np.eye(9)
        self.P = np.matmul(KHI, np.matmul(self.P, KHI.T))  # (KH-I)*P*(KH-I)'

        # add measurement noise and ensure boundedness and symmetry
        for i in range(9):
            for j in range(i,9):
                v = K[i] * R * K[j]
                p = 0.5*self.P[i][j] + 0.5*self.P[j][i] + v
                if np.isnan(p) or p>MAX_COV:
                    self.P[i][j] = MAX_COV
                    self.P[j][i] = MAX_COV
                elif i==j and p<MIN_COV:
                    self.P[i][j] = MIN_COV
                    self.P[j][i] = MIN_COV
                else:
                    self.P[i][j] = p
                    self.P[j][i] = p
                     

    def _updateWithTWR(self, twr: TWR_meas):
        """Update state with a TWR measurement

        Performs a measurement update with a single Two-Way Ranging
        Measurement.

        Args:
            twr: TWR_meas with which to update the state
        """

        dx = self.xi[STATE_X][0] - twr.anchor[0]
        dy = self.xi[STATE_Y][0] - twr.anchor[1]
        dz = self.xi[STATE_Z][0] - twr.anchor[2]

        d_meas = twr.distance
        d_pred = np.linalg.norm([dx,dy,dz])

        H = np.zeros((1,9))
        if d_pred != 0:
            H[0][STATE_X] = dx/d_pred
            H[0][STATE_Y] = dy/d_pred
            H[0][STATE_Z] = dz/d_pred
        else:
            H[0][STATE_X] = 1.0
            H[0][STATE_Y] = 0.0
            H[0][STATE_Z] = 0.0

        # outlier rejection based on Mahalonobis distance    
        error = d_meas-d_pred
        self._scalarUpdate(H, error, twr.stdDev, filterOutliers=True)

    def _updateWithTDOA(self, tdoa: TDOA_meas):
        """Update state with a TDOA measurement

        Performs a measurement update with a single Time-Difference of
        Arrival measurement.

        Args:
            tdoa: TDOA_meas with which to update the state
        """
        
        # Outlier rejection does not work properly for the first few 
        # measurements
        if self.tdoa_count > 20:
            meas = tdoa.distDiff
            pos = np.array([self.xi[STATE_X][0], self.xi[STATE_Y][0], self.xi[STATE_Z][0]])
            

            dx1 = pos - tdoa.anchorB
            dx0 = pos - tdoa.anchorA

            d1 = np.linalg.norm(dx1)
            d0 = np.linalg.norm(dx0)

            pred = d1-d0
            error = meas - pred

            H = np.zeros((1,9))
            if (d1 != 0) and (d0 != 0):
                h = dx1/d1 - dx0/d0
                H[0][STATE_X] = h[0]
                H[0][STATE_Y] = h[1]
                H[0][STATE_Z] = h[2]


            self._scalarUpdate(H, error, tdoa.stdDev, filterOutliers=True)
        
        self.tdoa_count += 1


    def _updateWithAlt(self, alt: Alt_meas):
        """Update state with an altitude measurement

        Performs a measurement update with a single altitude measurement.

        Args:
            alt: Alt_meas with which to update the state
        """

        if self.R[2][2]>0.1:
            angle = abs(np.cos(self.R[2][2])) - np.pi/180 * (15.0/2.0)
            if angle<0:
                angle = 0
            
            pred = self.xi[STATE_Z][0] / self.R[2][2]
            meas = alt.alt

            H = np.zeros((1,9))
            H[0][STATE_Z] = 1/self.R[2][2]

            self._scalarUpdate(H, meas-pred, alt.stdDev)


    def get_estimate_now(self, sim_time):
        """Returns current state estimate as State_XVQW"""
        
        est_x = [self.xi[STATE_X][0],self.xi[STATE_Y][0],self.xi[STATE_Z][0]]
        v_tmp = np.matmul(self.R, self.xi[STATE_PX:STATE_PZ+1])
        est_v = [v_tmp[0][0], v_tmp[1][0], v_tmp[2][0]]
        est_q = math3d.Quaternion([self.q[0], self.q[1], self.q[2], self.q[3]])
        
        return State_XVQW(sim_time, est_x, est_v, est_q, self.gyro)


def EKF_ASSERT_STATE_VALID(state, covariance):
    """Asserts if a state is valid

    Raises an AssertionError if the provided state has state variables
    that are out of bounds or NaN.

    Args:
        state: state to assess
        covariance: covariance matrix of the state

    Raises:
        AssertionError 
    """

    assert(not np.isnan(state).any())
    assert(not np.isnan(covariance).any())
    
    assert(abs(state[STATE_X])<100)
    assert(abs(state[STATE_Y])<100)
    assert(abs(state[STATE_Z])<100)
    assert(abs(state[STATE_PX])<10)
    assert(abs(state[STATE_PY])<10)
    assert(abs(state[STATE_PZ])<10)