"""UWBsim - Simulator for state estimation with UWB

UWBsim is a simulator for comparing state estimation techniques for
UAVs using Ultra Wide-band ranging. The simulation is based on log files
of a flown trajectory, but can generate UWB measurements with custom 
noise distributions. IMU data should however be provided by the log file.
"""
import os

# Set path variables for simulation files and data
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..','data')
PREFERENCE_FILE= os.path.join(BASE_DIR, '.preferences.yaml')