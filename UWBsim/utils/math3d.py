"""Utility functions and classes for 3D maths
"""

import numpy as np
import warnings

def RotationMatrix(roll, pitch, yaw):
    """Create corresponding rotation matrix

    Create rotation matrix from Tait-Bryan angles for a rotation order
    Yaw > Pitch > Roll (angles in local coordinates) 
    """
    sphi = np.sin(roll)
    cphi = np.cos(roll)
    stheta = np.sin(pitch)
    ctheta = np.cos(pitch)
    spsi = np.sin(yaw)
    cpsi = np.cos(yaw)

    R = np.zeros((3,3))

    R[0][0] = ctheta*cpsi
    R[0][1] = sphi*stheta*cpsi - cphi*spsi
    R[0][2] = sphi*spsi + cphi*stheta*cpsi

    R[1][0] = ctheta*spsi
    R[1][1] = cphi*cpsi + sphi*stheta*spsi
    R[1][2] = cphi*stheta*spsi - sphi*cpsi

    R[2][0] = -stheta
    R[2][1] = sphi*ctheta
    R[2][2] = cphi*ctheta

    return R


def rot2rpy(R: np.ndarray):
    """Get Tait-Bryan angles from rotation matrix

    Returns attitude as [roll, pitch, yaw], calculated from a Rotation Matrix R.
    Order of rotation: yaw > pitch > roll in local coordinates
    Attitude angles are between -pi and pi.
    """

    if R.shape != (3,3):
        raise TypeError('Rotation Matrix must be 3x3!')
    else:

        psi = np.arctan2(R[1][0],R[0][0])
        theta = np.arctan2(-R[2][0], np.sqrt(R[2][1]**2 + R[2][2]**2))
        phi = np.arctan2(R[2][1],R[2][2])
        return [phi, theta, psi]


class Quaternion(np.ndarray):
    """Quaternion class as subtype of np.ndarray

    Redefines multiplication for quaternion multiplication and defines 
    additional functions on ndarrays that are useful for quaternions

    Methods:
        v(): return the vector portion of the quaternion
        inv(): return the inverse quaternion
        get_att(): Convert quaternion to Tait-Bryan angles
        get_roll()/get_pitch()/get_yaw(): Only get individual Tait-Bryan angle
        rotation_matrix(): Convert quaternion to rotation matrix 
    """
    def __new__(subtype, q=[1,0,0,0]):
        try:
            q = np.array(q)
            #print('math q={}, {}'.format(q, type(q[2])))
            #assert(isinstance(q[2], (np.int64, np.float, int,float)))
            if q.shape != (4,):
                raise ValueError
        except ValueError:
            raise ValueError('Quaternion must have shape (4)')
        obj = super(Quaternion, subtype).__new__(subtype, q.shape, dtype=float, buffer=None, offset=0,
                                            strides=None, order=None)
        # Complex values can appear in eigenvalue calculation due to numerical
        # errors but can safely be ignored (symetric matrices should never have complex eigenvalues)
        warnings.simplefilter("ignore", np.ComplexWarning)
        obj[0] = q[0]
        obj[1] = q[1]
        obj[2] = q[2]
        obj[3] = q[3]
        return obj
    
    def __mul__(self, other):
        """Quaternion multiplication
        
        Treats 3D vectors as quaternions with a scalar part of 0
        """
        if isinstance(other, Quaternion):
            q = self
            p = other
            r0 = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
            r1 = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2]
            r2 = q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1] 
            r3 = q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]            
            return Quaternion([r0,r1,r2,r3])
        elif isinstance(other, (np.ndarray, list)):
            other = np.array(other)
            if other.shape == (4,):
                # treat as quaternion
                return self*Quaternion(other)
            elif other.shape ==(3,):
                # treat as 3D vector
                return self*Quaternion([0, other[0], other[1], other[2]])
            else:
                return NotImplemented    
        else:
            return super(Quaternion, self).__mul__(other)

    def v(self):
        """Return 3d vector if quaternion can be interpreted as such"""
        if self[0] < 0.0000001:
            return np.array([self[1], self[2], self[3]])
        else:
            raise AttributeError(f"Quaternion cannot be interpreted as 3d vector, q0={self[0]}")
    
    def inv(self):
        """Return the inverse of the quaternion """
        return Quaternion([self[0], -self[1], -self[2], -self[3]])

    def get_roll(self):
        """Calculate corresponding roll in Tait-Bryan angles"""
        q = self
        return np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]*q[1]+q[2]*q[2]))
    
    def get_pitch(self):
        """Calculate corresponding pitch in Tait-Bryan angles"""
        q = self
        return np.arcsin(2*(q[0]*q[2]-q[1]*q[3]))
    
    def get_yaw(self):
        """Calculate corresponding yaw in Tait-Bryan angles"""
        q = self
        return np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))
    
    def get_att(self):
        """Calculate corresponding Tait-Bryan angles"""
        return np.array([self.get_roll(), self.get_pitch(), self.get_yaw()])

    def rotation_matrix(self):
        """Calculate corresponding rotation matrix"""
        q = self
        R = np.zeros((3,3))
        R[0][0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
        R[0][1] = 2 * (q[1]*q[2] - q[0]*q[3])
        R[0][2] = 2 * (q[1]*q[3] + q[0]*q[2])

        R[1][0] = 2 * (q[1]*q[2] + q[0]*q[3])
        R[1][1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3]
        R[1][2] = 2 * (q[2]*q[3] - q[0]*q[1])

        R[2][0] = 2 * (q[1]*q[3] - q[0]*q[2])
        R[2][1] = 2 * (q[2]*q[3] + q[0]*q[1])
        R[2][2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
        return R

def aa2quat(angle, axis):
    "Calculate rotation quaternion from axis-angle representation."
    axis = axis/np.linalg.norm(axis)
    ca = np.cos(angle/2)
    sa = np.sin(angle/2)
    q = Quaternion([ca, sa*axis[0], sa*axis[1], sa*axis[2]])
    return q

def rpy2quat(roll, pitch, yaw):
    "Calculate rotation quaternion from roll, pitch, yaw."
    q_roll = aa2quat(roll, [1,0,0])
    q_pitch = aa2quat(pitch, [0,1,0])
    q_yaw = aa2quat(yaw, [0,0,1])

    return q_yaw*q_pitch*q_roll

def skew(v):
    """Returns the skew-symmetric matrix of vector v"""
    mat = np.array([[  0.0,-v[2], v[1]],
                    [ v[2],  0.0,-v[0]],
                    [-v[1], v[0], 0.0 ]])
    return mat