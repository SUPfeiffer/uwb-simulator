"""Collect data for comparison of MHE and EKF with heavy-tailed noise

This script simulates the performance of MHE and EKF on the trajectories
in the data/publication_run folder. Using a constant number of anchors,
UWB measurements are generated with different intensities of heavy-tailed
noise.
The position RMSE for both MHE and EKF is recorded in a csv file that
can then be used to generate plots with the 'publication_plots.py' script.
"""

import os
import yaml
import numpy as np

import UWBsim
from UWBsim.utils.uwb_ranging import RangingType, RangingSource
from UWBsim.simulation import UWBSimulation, SimulationParams

# Script settings
mode = 'twr'
runs_per_traj_file = 5
data_folder = os.path.join(UWBsim.DATA_DIR, 'publication_run')
anchor_file = os.path.join(UWBsim.BASE_DIR, 'anchor_positions.yaml')


# Set estimators parameters
params = SimulationParams()
params.estimators.mhe.enable = True
params.estimators.mhe.rate = 50
params.estimators.mhe.N_max = 20
params.estimators.mhe.iterations = 1
if mode=='twr':
    params.estimators.mhe.alpha = 0.02
else:
    params.estimators.mhe.alpha = 0.02
params.estimators.mhe.ransac_iterations = 10
params.estimators.mhe.ransac_fraction = 0.4
params.estimators.mhe.ransac_threshold = 1.7
params.estimators.ekf.enable = True
params.estimators.ekf.rate = 100

params.drone.altitude_enable = True

# Anchors
with open(anchor_file) as f:
    positions = yaml.safe_load(f)   
    params.ranging.anchor_positions = []  
    for key,pos in positions.items():
        i = int(key)
        params.ranging.anchor_positions.append([pos['x'], pos['y'], pos['z']])
params.ranging.anchor_enable = [False, True, False, True, 
                                True, False, True, False]

# Ranging parameters
if mode=='twr':
    params.ranging.interval = 0.05
    params.ranging.rtype = RangingType.TWR
    params.ranging.source = RangingSource.GENERATE_HT_GAMMA
    params.ranging.gauss_sigma = 0.1
    params.ranging.htg_lambda = 3.5
    params.ranging.htg_k = 2
    params.estimators.ekf.outlierThreshold = 1500

elif mode=='tdoa':
    params.ranging.interval = 0.1
    params.ranging.rtype = RangingType.TDOA
    params.ranging.source = RangingSource.GENERATE_HT_CAUCHY
    params.ranging.gauss_sigma = 0.3
    params.ranging.htc_gamma = 0.3
    params.estimators.ekf.outlierThreshold = 25

else:
    print('Invalid mode')
    exit()


# Create unique output file
output_file = 'publication/{}_ht_noise.csv'.format(mode)

i = 1
tmp = output_file
while os.path.isfile(tmp):
    tmp = output_file.split('.')[0] + str(i) + '.csv'
    i += 1
output_file = tmp

# Save settings for reference
settings_file = output_file.split('.')[0] + '_settings.yaml'
with open(settings_file, 'w') as f:
    yaml.dump(params, f)

# Global variables for error calculation
mhe_error_sum2 = np.array([0.0,0.0,0.0])
ekf_error_sum2 = np.array([0.0,0.0,0.0])
error_count = 0

def data_callback(drone):
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
    print('Writing to {}'.format(output_file))
    # Write output file header
    f_out.write('log, scale, run, mhe_tot, ekf_tot, mheX, mheY, mheZ, \
                    ekfX, ekfY, ekfZ, logfile\n')
    # iterate through all logs
    traj_names = []
    for f in os.listdir(data_folder):
        if f.startswith('.'):
            continue
        # Create unique readable name for trajectories
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

        if mode=='twr':
            scale_range = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        else:
            scale_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for scale in scale_range:
            if mode =='twr':
                params.ranging.htg_scale = scale
                params.ranging.htg_mu = 0.1*scale
            else:
                params.ranging.htc_ratio = scale
            for run in range(runs_per_traj_file):
                params.name = name + '_ht' + str(scale) + '_r' + str(run)
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

                mhe_tot = np.sqrt(mheX**2 + mheY**2 + mheZ**2)
                ekf_tot = np.sqrt(ekfX**2 + ekfY**2 + ekfZ**2)
                
                f_out.write('{}, {}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, \
                    {:.3f}, {:.3f}, {:.3f}, {:.3f}, {}\n'.format(
                    name, scale, run, mhe_tot, ekf_tot, mheX, mheY, 
                    mheZ, ekfX, ekfY, ekfZ, params.drone.logfile
                ))

