"""Implementation of Madgwick's and Mahony's IMU and AHRS algorithms

For more information see:
http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms

Adapted from the original MATLAB implementation by SOH Madgwick

Classes:
    Madgwick: Madgwick's AHRS algorithm
    Mahony: Mahony's AHRS algorithm
"""

import numpy as np
import UWBsim.utils.math3d as math3d

class Madgwick:
    """Madgwick's AHRS algorithm 
    
    Estimates attitude based on gyroscope and accelerometer measurements.
    Access current attitude quaternion via [Madgwick_instance].quat

    Methods:
        step(): calculate new attitude estimate based on new IMU measurements
    """

    def __init__(self, beta=1):
        self.quat = math3d.Quaternion([1.0, 0.0, 0.0, 0.0])
        self.beta = beta
        self.t_prev = 0

    def step(self, time, gyro, acc):
        """ Simulate the next time step
        
        Calculates new attitude quaternion from new IMU measurements

        Args:
            time: current time
            gyro: new gyroscope measurements (3D)
            acc: new accelerometer measurements (3D)
        """

        dt = time - self.t_prev
        q = self.quat

        if np.linalg.norm(acc) == 0:
            return
        else:
            # Normalize accelerometer measurement
            acc = acc/np.linalg.norm(acc)

            # Gradient decent algorithm corrective step
            F = [2*(q[1]*q[3] - q[0]*q[2]) - acc[0],
                 2*(q[0]*q[1] + q[2]*q[3]) - acc[1],
                 2*(0.5 - q[1]**2 - q[2]**2) - acc[2]]
            J = [[-2*q[2], 2*q[3],    -2*q[0],	2*q[1]],
                 [2*q[1],  2*q[0],     2*q[3],	2*q[2]],
                 [0,      -4*q[1],    -4*q[2],	0    ]]
            
            delta = np.matmul(np.transpose(J),F)
            if np.linalg.norm(delta) != 0:
                delta = delta/np.linalg.norm(delta)

            # compute rate of change of quaternion
            qdot = 0.5*q*gyro - self.beta*delta

            # integrate to yield quaternion
            q = q + qdot*dt
            self.quat = q/np.linalg.norm(q)

            # propagate time
            self.t_prev = time

class Mahony:
    """Mahony's AHRS algorithm 
    
    Estimates attitude based on gyroscope and accelerometer measurements.
    Access current attitude quaternion via [Madgwick_instance].quat

    Methods:
        step(): calculate new attitude estimate based on new IMU measurements
    """
    
    def __init__(self, kp=1, ki=0):
        self.quat = math3d.Quaternion([1,0,0,0])
        self.kp = kp
        self.ki = ki
        self.e_int = np.array([0,0,0])
        self.t_prev = 0

    def step(self, time, gyro, acc):
        """ Simulate the next time step
        
        Calculates new attitude quaternion from new IMU measurements

        Args:
            time: current time
            gyro: new gyroscope measurements (3D)
            acc: new accelerometer measurements (3D)
        """

        dt = time - self.t_prev
        
        q = self.quat

        if np.linalg.norm(acc) == 0:
            return
        else:
            # Normalise accelerometer measurement
            acc = acc / np.linalg.norm(acc)
 
            # Estimated direction of gravity and magnetic flux
            v = [2*(q[1]*q[3] - q[0]*q[2]),
                 2*(q[0]*q[1] + q[2]*q[3]),
                 q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]
            
            # Error is sum of cross product between estimated direction and measured direction of field
            e = np.cross(acc, v) 
            if self.ki > 0:
                self.e_int = self.e_int + e * dt   
            else:
                self.e_int = np.array([0, 0, 0])
            
            # Apply feedback terms
            gyro = gyro + self.kp * e + self.ki * self.e_int           
            
            # Compute rate of change of quaternion
            qDot = 0.5 * q * gyro
 
            # Integrate to yield quaternion
            q = q + qDot * dt
            self.quat = q / np.linalg.norm(q) # normalise quaternion

            # propagate time
            self.t_prev = time
