"""Dataclasses for custom datatypes

"""
import numpy as np
from dataclasses import dataclass, field
from typing import List

from UWBsim.utils.math3d import Quaternion

@dataclass
class State_XVQW:
    def __init__(self, timestamp: float=0.0, x: np.ndarray=np.zeros(3), 
                    v: np.ndarray=np.zeros(3), q: Quaternion=Quaternion([1.0,0.0,0.0,0.0]), w: np.ndarray=np.zeros(3)):
        self.timestamp = float(timestamp)
        self.x = np.array(x)
        self.v = np.array(v)
        if isinstance(q, Quaternion):
            self.q = q
        else:
            self.q = Quaternion([q[0],q[1],q[2],q[3]])
        self.w = np.array(w)

@dataclass
class TWR_meas:
    anchor: List[float]
    anchor_id: int
    distance: float
    stdDev: float = 0.25

    timestamp: float = 0.0

    def value(self):
        return self.distance

    def h(self, xi, u):
        return np.sqrt((self.anchor[0]-xi[0])**2 + (self.anchor[1]-xi[1])**2 + (self.anchor[2]-xi[2])**2)

    def gradh(self, xi, u):
        gh = np.zeros(10)
        gh[0:3] = np.array([self.anchor[0]-xi[0],self.anchor[1]-xi[1],self.anchor[2]-xi[2]])/self.h(xi,u)
        return gh

@dataclass
class TDOA_meas:
    anchorA: List[float]
    anchorB: List[float]
    anchorA_id: int
    anchorB_id: int
    distDiff: float
    stdDev: float = 0.25

    timestamp: float = 0.0
    def value(self):
        return self.distDiff

@dataclass
class Gyro_meas:
    rollRate: float
    pitchRate: float
    yawRate: float

    timestamp: float = 0.0

@dataclass
class Acc_meas:
    ax: float
    ay: float
    az: float

    timestamp: float = 0.0

@dataclass
class IMU_meas:
    gyro: List[float]
    acc: List[float]

    timestamp: float = 0.0

@dataclass
class Alt_meas:
    alt: float = 0.0
    stdDev: float = 0.1

    timestamp: float = 0.0

    def value(self):
        return self.alt