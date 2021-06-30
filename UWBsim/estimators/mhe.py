"""Moving Horizon Estimator for Position

This file contains the MHE for Position estimation as well as the 
datastructure for passing parameters to the MHE. The Position MHE 
employs a 6 dimensional state vector (x,y,z,vx,vy,vz). Attitude is 
estimated using a complementary filter and added to the state estimate 
that is returned by the step function.

The MHE implemented here has a output error structure (i.e. no process 
noise is considered) and uses computationally efficient single-iteration 
gradient descent methods for improving the state estimate at the 
beginning of a moving horizon. The returned state is the forward 
prediction of that state to the current time.

Classes:
    MHE_Params: The parameter data structure
    MHE: The Moving Horizon Estimator
"""

import numpy as np
import random
import yaml
import time as timing

from dataclasses import dataclass, field
from typing import List

from UWBsim.estimators.complementary import Madgwick, Mahony
from UWBsim.utils.dataTypes import State_XVQW, TWR_meas, TDOA_meas, Alt_meas
from UWBsim.utils.math3d import Quaternion

# State indices
ST_X = 0
ST_Y = 1
ST_Z = 2
ST_PX = 3
ST_PY = 4
ST_PZ = 5

# Input indices
IN_Q0 = 0
IN_Q1 = 1
IN_Q2 = 2
IN_Q3 = 3
IN_THRUST = 4

# constants
G = 9.813

class MHE_Params(yaml.YAMLObject):
    """Parameter Structure for the numerical position MHE

    Structure for passing and saving initialization parameters 
    for the numerical position MHE. Inherits from YAMLObject to
    allow saving and loading parameters from yaml files.
    
    Attributes:
        yaml_tag: tag for yaml representation
    """

    yaml_tag = u'!MHE_Params'
    
    def __init__(self, enable=False, rate=50, N_max=20, iterations=1, 
                    alpha=0.01, mu=1.0, ransac_iterations=10, 
                    ransac_threshold=2.0, ransac_fraction=0.4, 
                    attitude_rate=250):
        """ Initializes MHE_Params

            Default initialization parameters have been determined from
            experiments with Crazyflie logs.

            Args:
                enable: boolean, indicates if MHE shall be enabled
                rate: update rate of the MHE
                N_max: maximum horizon length
                iterations: number of iterations of the gradient descent 
                algorithm per update step
                alpha: step size for simple and conjugate gradient methods
                mu: prior weighting in the cost function
                ransac_iterations: number of ransac iterations, set to 0
                for no RANSAC
                ransac_threshold: maximum error per UWB measurement in 
                RANSAC
                ransac_fracion: fraction of UWB measurements used to 
                create a RANSAC estimate
                attitude_rate: attitude update rate (should be greater 
                or equal to rate)
        """

        self.enable = enable
        self.rate = rate
        self.N_max = N_max
        self.iterations = iterations
        self.mu = mu
        self.alpha = alpha
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
        self.ransac_fraction = ransac_fraction
        
        # not tunable in GUI
        self.attitude_rate = attitude_rate

    def __repr__(self):
        return "%s(enable=%r, rate=%r, N_max=%r, iterations=%r, alpha=%r, mu=%r, \
            ransac_iterations=%r, ransac_threshold=%r, ransac_fraction=%r, \
            attitude_rate=%r)" % (
            self.__class__.__name__, self.enable, self.rate, self.N_max, 
            self.iterations, self.alpha, self.mu, self.ransac_iterations, 
            self.ransac_threshold, self.ransac_fraction, self.attitude_rate
        )

class MHE:
    """Numerical MHE for state estimation on a drone (Position)

    Estimator uses 6 states (x,y,z,vx,vy,vz), attitude is estimated by 
    the means of a complementary filter (Mahony Algorithm) and also 
    returned by the step function. Usage: after initialization, use 
    addInputs() and addMeasurements() to add input/measurement data to 
    the buffer. Call step() to update the estimate at t-N and receive 
    the estimated state at t.

    Methods:
        reset(): reset the estimator
        step(): Execute one estimation step and return current state
        getEstimateNow(): Get the current state after the last update
        addInputs(): Add inputs to the MHE input accumulators
        addMeasurement(): Add a measurement to the measurement buffer
    """

    def __init__(self, mhe_params: MHE_Params, mass, K_aero, 
                    initial_state=State_XVQW()):
        """Initializes the numerical MHE for position

        Args:
            mhe_params: Instance of MHE_Params containing the estimator 
            parameters
            mass: Mass of the drone
            K_aero: 3x3 drag matrix
            initial_state: state at which the estimator is initialized
        """

        self.N = mhe_params.N_max       # Desired Horizon length
        self.dt = 1.0/mhe_params.rate
        self.attitude_dt = 1.0/mhe_params.attitude_rate

        
        self.iterations = mhe_params.iterations
        self.ransac_iterations = mhe_params.ransac_iterations
        self.ransac_fraction = mhe_params.ransac_fraction
        self.ransac_threshold = mhe_params.ransac_threshold
        
        self.mu = mhe_params.mu
        self.alpha = mhe_params.alpha

        self.m = mass
        self.K_aero = K_aero

        #self.estimator_compl = Madgwick(beta=0.01) #(beta=0.003)
        self.estimator_compl = Mahony(ki=0.001, kp=0.4) 
        self.reset(initial_state)


    def reset(self, initial_state):
        """Resets the estimator to the initial state

        Resets all internal variables to the same values as on startup 
        and sets the provided initial state as state.

        Args:
            initial_state: State_XVQW state to which to reset
        """

        self.N_now = 0                  # Current Horizon length
        self.xi_prior = np.zeros((6))   # estimate for xi @ t-N
        self.xi_prior[ST_X] = initial_state.x[0]
        self.xi_prior[ST_Y] = initial_state.x[1]
        self.xi_prior[ST_Z] = initial_state.x[2]

        self.xi_now = np.zeros((6))     # estimate for xi @ t
        self.xi_now[ST_X] = initial_state.x[0]
        self.xi_now[ST_Y] = initial_state.x[1]
        self.xi_now[ST_Z] = initial_state.x[2]
        
        # input and measurement arrays, wrap around once full
        self.u = [MHE_input() for _ in range(self.N)]
        self.alt_measurements = [[] for _ in range(self.N)]
        self.uwb_measurements = [[] for _ in range(self.N)]
        
        self.alt_buffer = []
        self.uwb_buffer = []
        self.tdoa_counter = 0

        self.isFlying = False
        self.i0 = 0     # index of oldest (i.e. t-N) value in wrapping arrays
        
        self.inputCounter = 0
        self.inputAccum_f = 0
        self.inputAccum_w = np.zeros(3)
        self.inputAccum_acc = np.zeros(3)
        self.inputAccum_Q = np.zeros((4,4))

        self.last_thrust = 0
        self.last_w = np.zeros(3)
        self.last_q = Quaternion([1,0,0,0])
        self.last_att_time = 0

    
    def f(self, xi, u):
        """Prediction equation

        Predicts the state at xi[k+1] based on the previous state and 
        input vector: xi[k+1] = f(xi[k], u[k])
        
        Args:
            xi: previous state (ndarray)
            u : previous input (MHE_input structure)
        
        Returns:
            new state xi[k+1] = f(xi,u)
        """

        xi_new = np.zeros(6)
        if not self.isFlying:
            xi_new += xi
        else:
            # x/y/z in body frame:
            dx = xi[ST_PX] * self.dt
            dy = xi[ST_PY] * self.dt
            dz = xi[ST_PZ] * self.dt + 0.5 * self.dt**2 *u.f_m

            # rotate to global frame
            dxi = np.zeros(6)

            dxi[ST_X] = dx + 2*((-u.q[2]*u.q[2] - u.q[3]*u.q[3]) * dx \
                               +( u.q[1]*u.q[2] - u.q[0]*u.q[3]) * dy \
                               +( u.q[0]*u.q[2] + u.q[1]*u.q[3]) * dz)
            dxi[ST_Y] = dy + 2*(( u.q[1]*u.q[2] + u.q[0]*u.q[3]) * dx \
                               +(-u.q[1]*u.q[1] - u.q[3]*u.q[3]) * dy \
                               +( u.q[2]*u.q[3] - u.q[0]*u.q[1]) * dz)
            dxi[ST_Z] = dz + 2*(( u.q[1]*u.q[3] - u.q[0]*u.q[2]) * dx \
                               +( u.q[2]*u.q[3] + u.q[0]*u.q[1]) * dy \
                               +(-u.q[1]*u.q[1] - u.q[2]*u.q[2]) * dz) \
                                   - 0.5*G*self.dt**2

            dxi[ST_PX] = -self.K_aero[0]*xi[ST_PX] + u.w[2]*xi[ST_PY] \
                         - u.w[1]*xi[ST_PZ] + 2*G*(u.q[0]*u.q[2]-u.q[1]*u.q[3])
            dxi[ST_PY] = -u.w[2]*xi[ST_PX] - self.K_aero[1]*xi[ST_PY] \
                         + u.w[0]*xi[ST_PZ] - 2*G*(u.q[2]*u.q[3]+u.q[0]*u.q[1])
            dxi[ST_PZ] = u.f_m + u.w[1]*xi[ST_PX] - u.w[0]*xi[ST_PY] \
                         - self.K_aero[2]*xi[ST_PZ] - G + 2*G*(u.q[1]*u.q[1]+u.q[2]*u.q[2])
            
            xi_new[ST_X] = xi[ST_X] + dxi[ST_X] 
            xi_new[ST_Y] = xi[ST_Y] + dxi[ST_Y] 
            xi_new[ST_Z] = xi[ST_Z] + dxi[ST_Z]
            xi_new[ST_PX] = xi[ST_PX] + self.dt*dxi[ST_PX]
            xi_new[ST_PY] = xi[ST_PY] + self.dt*dxi[ST_PY]
            xi_new[ST_PZ] = xi[ST_PZ] + self.dt*dxi[ST_PZ]

        MHE_ASSERT_STATE_VALID(xi_new)
        return xi_new
    

    def Df(self, xi, u):
        """Jacobian of the prediction equation

        Calculates the Jacobian of the prediction equation based on xi
        and u
        
        Args:
            xi: previous state (ndarray)
            u : previous input (MHE_input structure)
        
        Returns:
            Jacobian Df = df/dxi(xi,u)
        """

        Df = np.zeros((6,6))

        # dfx/dx
        Df[ST_X][ST_X] = 1
        Df[ST_Y][ST_Y] = 1
        Df[ST_Z][ST_Z] = 1

        # dfx/drho
        Df[ST_X][ST_PX] = 2 * self.dt * (0.5 - u.q[2]*u.q[2] - u.q[3]*u.q[3])
        Df[ST_X][ST_PY] = 2 * self.dt * (u.q[1]*u.q[2] - u.q[0]*u.q[3])
        Df[ST_X][ST_PZ] = 2 * self.dt * (u.q[0]*u.q[2] + u.q[1]*u.q[3])

        Df[ST_Y][ST_PX] = 2 * self.dt * (u.q[1]*u.q[2] + u.q[0]*u.q[3])
        Df[ST_Y][ST_PY] = 2 * self.dt * (0.5 - u.q[1]*u.q[1] - u.q[3]*u.q[3])
        Df[ST_Y][ST_PZ] = 2 * self.dt * (u.q[2]*u.q[3] - u.q[0]*u.q[1])

        Df[ST_Z][ST_PX] = 2 * self.dt * (u.q[1]*u.q[3] - u.q[0]*u.q[2])
        Df[ST_Z][ST_PY] = 2 * self.dt * (u.q[2]*u.q[3] + u.q[0]*u.q[1])
        Df[ST_Z][ST_PZ] = 2 * self.dt * (0.5 - u.q[1]*u.q[1] - u.q[2]*u.q[2])

        # dfrho/drho
        Df[ST_PX][ST_PX] = 1 - self.dt * self.K_aero[0]
        Df[ST_PX][ST_PY] = u.w[2] * self.dt
        Df[ST_PX][ST_PZ] = -u.w[1] * self.dt

        Df[ST_PY][ST_PX] = -u.w[2] * self.dt
        Df[ST_PY][ST_PY] = 1 - self.dt * self.K_aero[1]
        Df[ST_PY][ST_PZ] = u.w[0] * self.dt
        
        Df[ST_PZ][ST_PX] = u.w[1] * self.dt
        Df[ST_PZ][ST_PY] = -u.w[0] * self.dt
        Df[ST_PZ][ST_PZ] = 1 - self.dt * self.K_aero[2]

        return Df


    def step(self, time): 
        """ Execute one estimation step of the MHE and return current estimate

        Shifts the moving horizon and moves new inputs and measurements 
        from buffers into internal variables. Then updates the estimated
        state at t-N and returns the estimate of the current state.

        Args:
            time: current time of the simulation
        
        Returns:
            Current state estimate at 'time' in XVQW_State format
        
        Raises:
            AssertionError: State estimate is out of bounds or NaN
        """

        u_tmp = self._get_inputs_from_accumulators(time)
        if self.N_now < self.N: 
            # Increase horizon length (append inputs/measurements)
            i = self.N_now
            self.u[i] = u_tmp
            self.uwb_measurements[i] = self.uwb_buffer
            self.alt_measurements[i] = self.alt_buffer
            self.N_now += 1
        else:
            # Update prior and overwrite oldest inputs/measurements
            i = self.i0
            self.xi_prior = self.f(self.xi_prior, self.u[i])
            self.u[i] = u_tmp
            self.uwb_measurements[i] = self.uwb_buffer
            self.alt_measurements[i] = self.alt_buffer
            self.i0 += 1
            if self.i0 >= self.N:
                self.i0 = 0

        self.uwb_buffer = []       
        self.alt_buffer = []

        self.xi_prior = self._gradient_method(self.xi_prior)
        state_estimate_now = self.getEstimateNow(time)
        
        return state_estimate_now


    def getEstimateNow(self, time):
        """Returns the state estimate at the time of the last estimator update

        Predicts the current state from the estimated state at t-N and 
        the sequence of inputs saved by the MHE. The result is returned 
        in a State_XVQW object.

        Args:
            time: timestamp for the estimate
        
        Returns:
            State_XVQW object with the state at time of the last 
            estimator update
        """
        
        state = np.zeros(6)
        state += self.xi_prior 

        for i in range(self.N_now-1):
            idx = (self.i0 + i) % self.N    
            state = self.f(state, self.u[idx])

        est_x = [state[ST_X], state[ST_Y], state[ST_Z]]
        est_q = self.last_q
        est_v = (est_q*[state[ST_PX],state[ST_PY],state[ST_PZ]]*est_q.inv()).v()
        est_w = self.last_w
        
        return State_XVQW(time, est_x, est_v, est_q, est_w)


    def addInputs(self, thrust, gyro, acc, time):
        """Add inputs to MHE input accumulators

        Adds inputs to the MHE input accumulators which act as buffer 
        between steps of the estimator. The accumulators are emptied by 
        the MHE step function, which averages the inputs to a single 
        value (per input) per timestep. 

        Args:
            thrust: thrust measurement in N, float
            gyro: 3D gyro measurements in radians, list(3)
            acc: 3D accelerometer measurement in m/s2, list(3)
            time: current time, float
        """

        if not self.isFlying and thrust>=0.99*G*self.m:
            self.isFlying = True
        
        if (time - self.last_att_time > self.attitude_dt):
            self.estimator_compl.step(time, gyro, acc)
            q = self.estimator_compl.quat
            self.inputAccum_Q += np.array([q[0]*q, q[1]*q, q[2]*q, q[3]*q])
            self.last_att_time = time
            self.last_q = q

        self.inputCounter += 1
        self.inputAccum_f += thrust
        self.inputAccum_w += gyro
        self.inputAccum_acc += acc
        self.last_thrust = thrust
        self.last_w = gyro


    def _get_inputs_from_accumulators(self, time):
        """ Empty MHE input accumulators and return scaled MHE input object
        
        Uses the MHE input accumulators and the MHE input counter to 
        average the accumulated inputs since the last function execution 
        and resets them. Average inputs are scaled and returned in an 
        MHE input object.

        Args:
            time: current time
        Returns:
            A MHE input object containing averaged inputs since last 
            function execution
        """

        u = MHE_input()

        # Timestamp at t-dt, because inputs are collected between 
        # xi[t-dt] and xi[t], and the state prediction should be
        # xi[t] = f(xi[t-dt], u[t-dt])
        u.timestamp = time - self.dt 

        if self.inputCounter > 0:
            avg_f = self.inputAccum_f / self.inputCounter
            avg_w = self.inputAccum_w / self.inputCounter
            avg_acc = self.inputAccum_acc / self.inputCounter

            # Calculate average quaternion, details in 
            # http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
            ew, ev = np.linalg.eig(self.inputAccum_Q)
            idx = np.argmax(ew)        
            avg_q = np.array([ev[0][idx], ev[1][idx], ev[2][idx], ev[3][idx]])

            u.f_m = avg_f / self.m
            u.w = np.array(avg_w)
            u.a = np.array(avg_acc)
            u.q = Quaternion(avg_q)

        # Reset input accumulators and counters
        self.inputCounter = 0
        self.inputAccum_f = 0
        self.inputAccum_w = np.zeros(3)
        self.inputAccum_acc = np.zeros(3)
        self.inputAccum_Q = np.zeros((4,4))

        return u


    def addMeasurement(self, measurement):
        """Add a measurement to the MHE measurement buffer

        Adds any measurement to the MHE measurement buffer. The buffer 
        is transferred to the measurement list with every estimator step.
        This function does not enforce thetype of measurement that can 
        be added, but only measurements which are covered by the MHE 
        _get_h() and _get_Dh() will work with the estimator (also _get_D2h 
        for Newton Method). 

        Args:
            measurement: any type of measurement that can be used by the
            estimator
        """
        
        if isinstance(measurement, Alt_meas):
            self.alt_buffer.append(measurement)
        elif isinstance(measurement, TWR_meas):
            self.uwb_buffer.append(measurement)
        elif isinstance(measurement, TDOA_meas):
            if self.tdoa_counter > 10:
                self.uwb_buffer.append(measurement)
            else:
                self.tdoa_counter +=1
        else:
            print("[MHE] addMeasurement: Unknown measurement of type {}".format(type(measurement)))


    def _gradient_method(self, prior):
        """Calculates state estimate with simple gradient method

        Calculates an improved estimate at t-N by minimizing the MHE
        cost function using the simple gradient method. Parameters for 
        the method are set during initialization of the estimator class.
        If the number of RANSAC iterations is >0, RANSAC will be used to
        remove outliers in the UWB measurements. If enabled, instead of
        a fixed step size, a linesearch can be performed to improve
        convergence.

        Args:
            prior: Prior estimate of the state at t-N
        
        Returns:
            Improved state estimate at t-N
        """

        xi_hat = np.array(prior)
        for _ in range(self.iterations):
            f_i = xi_hat
            Df_i = np.eye(6)
            gradient_sum = np.zeros(6)
            uwb_gradient_terms = []
            # Calculate individual measurement cost contributions
            for i in range(self.N_now):
                i_now = (self.i0 + i) % self.N
                if i != 0:
                    # because we collect u after it has affected the system,
                    # the prediction equation is xi_{k+1} = f(xi_k, u_{k+1})
                    Df_i = np.matmul(Df_i, self.Df(f_i, self.u[i_now]))
                    f_i = self.f(f_i, self.u[i_now])

                for m in self.alt_measurements[i_now]:
                    h = self._get_h(m, f_i, self.u[i_now])
                    Dh = self._get_Dh(m, f_i, self.u[i_now])
                    Dh_cr = np.matmul(Dh,Df_i)
                    tmp = (m.value() - h)*np.transpose(Dh_cr)
                    tmp = np.transpose(tmp)
                    gradient_sum += tmp[0]
                    

                for m in self.uwb_measurements[i_now]: 
                    h = self._get_h(m, f_i, self.u[i_now])
                    Dh = self._get_Dh(m, f_i, self.u[i_now])
                    Dh_cr = np.matmul(Dh,Df_i)
                    tmp = (m.value() - h)*np.transpose(Dh_cr)
                    tmp = np.transpose(tmp)
                    uwb_gradient_terms.append(tmp)

            # RANSAC
            N_m = len(uwb_gradient_terms)
            N_smpl = int(np.ceil(self.ransac_fraction * N_m))
            N_smpl = max(2, N_smpl)

            if (self.ransac_iterations <= 0) or (N_smpl>N_m):
                # no RANSAC, use all measurements
                for cost in uwb_gradient_terms:
                    gradient_sum += cost[0]
                gradJ = 2*self.mu*(xi_hat-prior) - 2*gradient_sum

                xi_hat = xi_hat - self.alpha*gradJ

            else:
                # use RANSAC
                best_error = 1000000000 # N_m*self.ransac_threshold + 1
                best_xi_hat = np.array(prior)
                best_inliers = np.ones(N_m)
                for _ in range(self.ransac_iterations):
                    # Calculate estimate from sample
                    cost_smpl = random.sample(uwb_gradient_terms, N_smpl)
                    meas_sum_smpl = np.sum(cost_smpl, axis=0)[0]
                    gradJ = 2*self.mu*(xi_hat-prior) - \
                                2*(gradient_sum + meas_sum_smpl)

                    xi_hat_smp = xi_hat - self.alpha*gradJ

                    # Evaluate estimate
                    f_i = xi_hat_smp
                    tot_error = 0
                    # smpl_error = np.zeros(N_m)
                    smpl_inliers = np.ones(N_m)
                    uwb_cnt = 0
                    for i in range(self.N_now):
                        if tot_error>best_error:
                            break
                        
                        i_now = (self.i0 + i) % self.N
                        if i != 0:
                            try:
                                f_i = self.f(f_i, self.u[i_now])
                            except AssertionError:
                                tot_error = best_error + 1
                                break
                        
                        for m in self.alt_measurements[i_now]:
                            h = self._get_h(m, f_i, self.u[i_now])
                            e = abs(m.value() - h)
                            tot_error += e

                        for m in self.uwb_measurements[i_now]:
                            h = self._get_h(m, f_i, self.u[i_now])
                            e = abs(m.value() - h)
                            # bound errors on uwb outliers
                            if e>self.ransac_threshold:
                                e = self.ransac_threshold
                                smpl_inliers[uwb_cnt] = 0
                            tot_error += e
                            uwb_cnt += 1

                    if tot_error < best_error:
                        best_error = tot_error
                        best_xi_hat = np.array(xi_hat_smp)
                        best_inliers = smpl_inliers
                
                # recalculate with all inliers from best solution
                best_uwb_measurements = [[] for _ in range(self.N)]
                uwb_cnt = 0
                for i in range(self.N_now):
                    i_now = (self.i0 + i) % self.N
                    for m in self.uwb_measurements[i_now]:
                        if best_inliers[uwb_cnt] != 0:
                            gradient_sum += uwb_gradient_terms[uwb_cnt][0]
                            best_uwb_measurements[i_now].append(m)
                        uwb_cnt += 1
              

                gradJ = 2*self.mu*(xi_hat-prior) - 2*gradient_sum

                xi_hat = xi_hat - self.alpha*gradJ
                

            MHE_ASSERT_STATE_VALID(xi_hat)
        
        return xi_hat


    def _get_h(self, m, xi, u):
        """Returns the expected value for a measurement
        
        This function returns the expected value of a measurement
        based on a state and input. This function choses the correct
        get_h() function depending on the type of the measurement 
        provided.

        Args:
            m: measurement of desired type
            xi: state vector
            u: input structure
        
        Returns:
            Value of the measurement function h_m(xi, u)
        """

        arg = type(m)
        switcher = {
            TWR_meas: self._h_twr,
            TDOA_meas: self._h_tdoa,
            Alt_meas: self._h_alt,
        }
        func = switcher.get(arg)
        return func(m, xi, u)

    def _h_twr(self, m, xi, u):
        """Measurement function for twr measurements """
        return np.sqrt((m.anchor[0]-xi[0])**2 + (m.anchor[1]-xi[1])**2 + (m.anchor[2]-xi[2])**2)

    def _h_tdoa(self, m, xi, u):
        """Measurement function for tdoa measurements """
        d1 = np.sqrt((xi[0] - m.anchorB[0])**2 + (xi[1] - m.anchorB[1])**2 + (xi[2] - m.anchorB[2])**2)
        d0 = np.sqrt((xi[0] - m.anchorA[0])**2 + (xi[1] - m.anchorA[1])**2 + (xi[2] - m.anchorA[2])**2)
        return d1-d0

    def _h_alt(self, m, xi, u):
        """Measurement function for altitude measurements """
        return xi[ST_Z]


    def _get_Dh(self, m, xi, u):
        """Returns the gradient of a measurement function
        
        This function returns the value of a measurement function
        based on a state and input. This function choses the correct
        _get_Dh() function depending on the type of the measurement 
        provided.

        Args:
            m: measurement of desired type
            xi: state vector
            u: input structure
        
        Returns:
            Gradient of the measurement function Dh_m(xi, u)
        """

        arg = type(m)
        switcher = {
            TWR_meas: self._Dh_twr,
            TDOA_meas: self._Dh_tdoa,
            Alt_meas: self._Dh_alt,
        }
        func = switcher.get(arg)
        return func(m, xi, u)


    def _Dh_twr(self, m, xi, u):
        """Measurement gradient for twr measurements """
        Dh = np.zeros((1,6))
        h = self._h_twr(m,xi,u)
        Dh[0][ST_X] = (xi[0] - m.anchor[0])/h
        Dh[0][ST_Y] = (xi[1] - m.anchor[1])/h
        Dh[0][ST_Z] = (xi[2] - m.anchor[2])/h
        return Dh

    def _Dh_tdoa(self, m, xi, u):
        """Measurement gradient for tdoa measurements """
        Dh = np.zeros((1,6))

        d1_vec = [xi[0] - m.anchorB[0], xi[1] - m.anchorB[1], xi[2] - m.anchorB[2]] 
        d0_vec = [xi[0] - m.anchorA[0], xi[1] - m.anchorA[1], xi[2] - m.anchorA[2]] 
        d1 = np.sqrt(d1_vec[0]**2 + d1_vec[1]**2 + d1_vec[2]**2)
        d0 = np.sqrt(d0_vec[0]**2 + d0_vec[1]**2 + d0_vec[2]**2)

        Dh[0][ST_X] = d1_vec[0]/d1 - d0_vec[0]/d0
        Dh[0][ST_Y] = d1_vec[1]/d1 - d0_vec[1]/d0
        Dh[0][ST_Z] = d1_vec[2]/d1 - d0_vec[2]/d0
        
        return Dh

    def _Dh_alt(self, m, xi, u):
        """Measurement gradient for altitude measurements """
        Dh = np.zeros((1,6))
        Dh[0][ST_Z] = 1
        return Dh

@dataclass
class MHE_input:
    """Data class for inputs to the MHE"""
    f_m:            float = 0.0
    w:              np.ndarray = field(default_factory=lambda: np.zeros(3))
    a:              np.ndarray = field(default_factory=lambda: np.zeros(3))
    q:              np.ndarray = field(default_factory=lambda: np.array([1.0,0.0,0.0,0.0]))
    timestamp:      float = 0.0

def MHE_ASSERT_STATE_VALID(state):
    """Asserts if a state is valid

    Raises an AssertionError if the provided state has state variables
    that are out of bounds or NaN.

    Args:
        state: state to assess

    Raises:
        AssertionError 
    """
    assert(not any(np.isnan(state)))
    
    assert(abs(state[ST_X])<100)
    assert(abs(state[ST_Y])<100)
    assert(abs(state[ST_Z])<100)
    assert(abs(state[ST_PX])<10)
    assert(abs(state[ST_PY])<10)
    assert(abs(state[ST_PZ])<10)