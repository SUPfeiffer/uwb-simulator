"""Collect data for comparison of MHE and EKF with different number of anchors

This script simulates the performance of MHE and EKF on the trajectories
in the data/publication_run folder. For each file, the number of anchors 
is varied between 1-8 for TWR and 2-8 for TDOA. Every number of anchor 
is tested in 10 runs with the anchors chosen randomly for every run.
The position RMSE for both MHE and EKF is recorded in a csv file that
can then be used to generate plots with the 'publication_plots.py' script.
"""

import os
import yaml
import numpy as np
import random

import UWBsim
from UWBsim.airframe.drone import Drone
from UWBsim.utils.uwb_ranging import RangingType, RangingSource
from UWBsim.simulation import UWBSimulation, SimulationParams


# Script settings
runs_per_traj_file = 10
mode = 'twr'
data_folder = os.path.join(UWBsim.DATA_DIR, 'publication_run')
anchor_file = os.path.join(UWBsim.BASE_DIR, 'anchor_positions.yaml')
publication_folder = os.path.dirname(os.path.realpath(__file__))


# Set Estimator parameters
params = SimulationParams()
params.estimators.mhe.enable = True
params.estimators.mhe.rate = 50
params.estimators.mhe.N_max = 20
params.estimators.mhe.iterations = 1
params.estimators.mhe.ransac_iterations = 10
params.estimators.mhe.ransac_fraction = 0.4
params.estimators.mhe.ransac_threshold = 1.7
params.estimators.mhe.mu = 10
params.estimators.ekf.enable = True
params.estimators.ekf.rate = 100

params.drone.altitude_enable = True

if mode == 'twr':
    n_anchors = [1, 2, 3, 4, 5, 6, 7, 8]
    params.ranging.rtype = RangingType.TWR
    mhe_alphas = [0.02, 0.02, 0.02, 0.015, 0.015, 0.015, 0.015, 0.015]
    params.estimators.ekf.outlierThreshold = 1500
elif mode == 'tdoa':
    n_anchors = [2, 3, 4, 5, 6, 7, 8]
    params.ranging.rtype = RangingType.TDOA
    mhe_alphas = [0.02, 0.02, 0.015, 0.015, 0.01, 0.01, 0.01]
    params.estimators.ekf.outlierThreshold = 25

with open(anchor_file) as f:
    positions = yaml.safe_load(f)
    params.ranging.anchor_positions = []
    for key,pos in positions.items():
        i = int(key)
        params.ranging.anchor_positions.append([pos['x'], pos['y'], pos['z']])

params.ranging.source = RangingSource.LOG


# Create unique output file
output_file = os.path.join(publication_folder, '{}_anchors.csv'.format(mode))

i = 1
tmp = output_file
while os.path.isfile(tmp):
    tmp = output_file.split('.')[0] + str(i) + '.csv'
    i += 1
output_file = tmp


# Save parameters for later reference
settings_file = output_file.split('.')[0] + '_settings.yaml'
with open(settings_file, 'w') as f:
    yaml.dump(params, f)


# Global variables for error calculation
mhe_error_sum2 = np.array([0.0,0.0,0.0])
ekf_error_sum2 = np.array([0.0,0.0,0.0])
error_count = 0


def data_callback(drone: Drone):
    """Record the simulation output in the scripts global variables

    This function is passed to the simulation and is called at every 
    simulation step. It records the true and estimated states of MHE
    and EKF in the scripts global variables, so that the performance
    can be calculated at the end of the simulation. 
    """
    global error_count, mhe_error_sum2, ekf_error_sum2
    # wait a moment before starting error calculation (calibration)
    if drone.time > 1.0:
        x = drone.state_true.x[0]
        y = drone.state_true.x[1]
        z = drone.state_true.x[2]
        
        error_count += 1

        if drone.estimator_isEnabled['mhe']:
            mhe_error_sum2[0] += (x - drone.state_estimate['mhe'].x[0])**2
            mhe_error_sum2[1] += (y - drone.state_estimate['mhe'].x[1])**2
            mhe_error_sum2[2] += (z - drone.state_estimate['mhe'].x[2])**2
        if drone.estimator_isEnabled['ekf']:
            ekf_error_sum2[0] += (x - drone.state_estimate['ekf'].x[0])**2
            ekf_error_sum2[1] += (y - drone.state_estimate['ekf'].x[1])**2
            ekf_error_sum2[2] += (z - drone.state_estimate['ekf'].x[2])**2

with open(output_file, 'w') as f_out:
    print('Writing to: {}'.format(output_file))
    # Write output file header
    f_out.write('log, anchors, run, mhe_tot, ekf_tot, mheX, mheY, mheZ, ekfX, \
        ekfY, ekfZ, logfile\n')
    # iterate through all logs
    traj_names = []
    for f in os.listdir(data_folder):
        if not mode in f:
            continue
        elif 'BAD' in f: # Exclude logs that lose UWB connection
            continue
        elif f.startswith('.'):
            continue

        # Create human readable name for trajectories
        name = f.split('.')[0]
        name = name.split('+')[-1]
        if 'twr' in f:
            name = 'twr_' + name
        elif 'tdoa' in f:
            name = 'tdoa_' + name
        
        letter=65 # chr(65)=A, chr(66)=B, ...
        while True:
            tmp = name + '_' + chr(letter)
            if tmp in traj_names:
                letter += 1
            else:
                name = tmp
                traj_names.append(name)
                break

       # Set Logfile for run
        params.drone.logfile = os.path.join(data_folder,f)
    
        for idx, Na in enumerate(n_anchors): 
            params.estimators.mhe.alpha = mhe_alphas[idx]

            for run in range(runs_per_traj_file):
                params.ranging.anchor_enable = [False, False, False, False,
                                                 False, False, False, False]
                anchor_idx_en = random.sample(range(8), Na)
                for a_idx in anchor_idx_en:
                    params.ranging.anchor_enable[a_idx] = True

                params.name = name + '_a' + str(Na) + '_r' + str(run)
                # Reset error calculation
                error_count = 0
                mhe_error_sum2[0] = 0
                mhe_error_sum2[1] = 0
                mhe_error_sum2[2] = 0
                ekf_error_sum2[0] = 0
                ekf_error_sum2[1] = 0
                ekf_error_sum2[2] = 0

                # Run simulation
                sim = UWBSimulation(params, NotImplemented, data_callback)
                try:
                    sim.start_sim()
                    mheX = np.sqrt(mhe_error_sum2[0]/error_count)
                    mheY = np.sqrt(mhe_error_sum2[1]/error_count)
                    mheZ = np.sqrt(mhe_error_sum2[2]/error_count)
                    ekfX = np.sqrt(ekf_error_sum2[0]/error_count)
                    ekfY = np.sqrt(ekf_error_sum2[1]/error_count)
                    ekfZ = np.sqrt(ekf_error_sum2[2]/error_count)
                except AssertionError:
                    # One of the estimators failed, try both individually
                    # MHE only
                    params.estimators.ekf.enable = False
                    error_count = 0
                    mhe_error_sum2[0] = 0
                    mhe_error_sum2[1] = 0
                    mhe_error_sum2[2] = 0
                    try:
                        sim = UWBSimulation(params, NotImplemented, 
                                data_callback)
                        sim.start_sim()
                        mheX = np.sqrt(mhe_error_sum2[0]/error_count)
                        mheY = np.sqrt(mhe_error_sum2[1]/error_count)
                        mheZ = np.sqrt(mhe_error_sum2[2]/error_count)

                    except AssertionError:
                        mheX = np.inf
                        mheY = np.inf
                        mheZ = np.inf
                    
                    finally:
                        params.estimators.ekf.enable = True
                    
                    # EKF only
                    params.estimators.mhe.enable = False
                    error_count = 0
                    ekf_error_sum2[0] = 0
                    ekf_error_sum2[1] = 0
                    ekf_error_sum2[2] = 0
                    try:
                        sim = UWBSimulation(params, NotImplemented, 
                                data_callback)
                        sim.start_sim()
                        ekfX = np.sqrt(ekf_error_sum2[0]/error_count)
                        ekfY = np.sqrt(ekf_error_sum2[1]/error_count)
                        ekfZ = np.sqrt(ekf_error_sum2[2]/error_count)

                    except AssertionError:
                        ekfX = np.inf
                        ekfY = np.inf
                        ekfZ = np.inf

                    finally:
                        params.estimators.mhe.enable = True

                # Calculate performance and write to output file
                mhe_tot = np.sqrt(mheX**2 + mheY**2 + mheZ**2)
                ekf_tot = np.sqrt(ekfX**2 + ekfY**2 + ekfZ**2)
                
                f_out.write('{}, {}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, \
                    {:.3f}, {:.3f}, {:.3f}, {:.3f}, {}\n'.format(
                    name, Na, run, mhe_tot, ekf_tot, mheX, mheY, 
                    mheZ, ekfX, ekfY, ekfZ, params.drone.logfile
                ))

