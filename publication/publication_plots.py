"""Script for the creation of plots for publication

This script creates plots for comparing MHE and EKF performance from
csv files that were created with the headless publication simulations.
Functions:
    plot_heavytail: Plot for comparison of simulation on heavy-tailed noise
    plot_anchors: Plot for comparison of different number of anchors
    plot_noise_twr: Plot noise model used for simulated TWR ranging
    plot_noise_tdoa: Plot noise model used for simulated TdoA ranging
"""

import csv
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Set parameters for plot fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

def plot_heavytail(file_name):
    """ Plot the comparison of EKF and MHE with simulated heavy-tailed noise
    """

    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))

    if 'twr' in file_name:
        scale = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    else:
        scale = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    N_scale = len(scale)

    mhe = [[] for _ in range(N_scale)]
    ekf = [[] for _ in range(N_scale)]
    
    mhe_out = [0 for _ in range(N_scale)]
    ekf_out = [0 for _ in range(N_scale)]

    mhe_in = [0 for _ in range(N_scale)]
    ekf_in = [0 for _ in range(N_scale)]

    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            for i in range(N_scale):
                if float(row[' scale']) == scale[i]:
                    # identify unsuccessful localizations
                    if float(row[' mhe_tot']) < 1:
                        mhe[i].append(float(row[' mhe_tot']))
                        mhe_in[i] += 1
                    else:
                        mhe_out[i] += 1
                    
                    if float(row[' ekf_tot']) < 1:
                        ekf[i].append(float(row[' ekf_tot']))
                        ekf_in[i] += 1
                    else:
                        ekf_out[i] += 1
                    

    print('mhe_out = {}'.format(mhe_out))
    print('ekf_out = {}'.format(ekf_out))
    mhe_avg = np.zeros(N_scale)
    ekf_avg = np.zeros(N_scale)

    mhe_stderror = np.zeros(N_scale)
    ekf_stderror = np.zeros(N_scale)

    mhe_success = np.zeros(N_scale)
    ekf_success = np.zeros(N_scale)

    for i in range(N_scale):
        mhe_avg[i] = np.mean(mhe[i])
        ekf_avg[i] = np.mean(ekf[i])

        mhe_stderror[i] = np.std(mhe[i])/np.sqrt(len(mhe[i]))
        ekf_stderror[i] = np.std(ekf[i])/np.sqrt(len(ekf[i]))

        mhe_success[i] = 100*mhe_in[i]/(mhe_in[i]+mhe_out[i])
        ekf_success[i] = 100*ekf_in[i]/(ekf_in[i]+ekf_out[i])

    fig, ax = plt.subplots()
    ax.plot(scale, ekf_avg, '.-')
    ax.fill_between(scale, ekf_avg-ekf_stderror, ekf_avg+ekf_stderror, alpha=0.3)
    ax.plot(scale, mhe_avg, '.-')
    ax.fill_between(scale, mhe_avg-mhe_stderror, mhe_avg+mhe_stderror, alpha=0.3)
    if 'twr' in file_name:
        plt.ylim((0,0.625))
    elif 'tdoa' in file_name:
        plt.ylim((0,0.625))
        axOut = ax.twinx()
        axOut.set_ylabel(r'Success rate (RMSE$<$1\,m) [\%]')
        plt.ylim((0,125))
        axOut.plot(scale, ekf_success, 'x--')
        axOut.plot(scale, mhe_success, 'x--')

    ax.grid(True)
    if 'twr' in file_name:
        ax.set( xlabel=r'Heavy-Tail scale factor $s$\textsubscript{ht}', ylabel='mean RMSE [m]')
    else:
        ax.set( xlabel=r'Heavy-Tail ratio $r$\textsubscript{ht}', ylabel='mean RMSE [m]')
    ax.legend(["EKF", "MHE"], loc='lower right')


def plot_anchors(file_name):
    """Plot the comparison of EKF and MHE for different number of anchors
    """
    
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))

    if 'twr' in file_name:
        anchors = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    else:
        anchors = np.array([2,3,4,5,6,7,8])
    N_anchors = len(anchors)

    mhe = [[] for _ in range(N_anchors)]
    ekf = [[] for _ in range(N_anchors)]
    
    mhe_out = [0 for _ in range(N_anchors)]
    ekf_out = [0 for _ in range(N_anchors)]


    mhe_in = [0 for _ in range(N_anchors)]
    ekf_in = [0 for _ in range(N_anchors)]

    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            for i in range(N_anchors):
                if int(row[' anchors']) == anchors[i]:
                    # Identify unsuccessful localizations
                    if float(row[' mhe_tot']) < 1:
                        mhe[i].append(float(row[' mhe_tot']))
                        mhe_in[i] += 1
                    else:
                        mhe_out[i] += 1
                    
                    if float(row[' ekf_tot']) < 1:
                        ekf[i].append(float(row[' ekf_tot']))
                        ekf_in[i] += 1
                    else:
                        ekf_out[i] += 1

    print('mhe_out = {}'.format(mhe_out))
    print('ekf_out = {}'.format(ekf_out))
    mhe_avg = np.zeros(N_anchors)
    ekf_avg = np.zeros(N_anchors)

    mhe_stderror = np.zeros(N_anchors)
    ekf_stderror = np.zeros(N_anchors)
    
    mhe_success = np.zeros(N_anchors)
    ekf_success = np.zeros(N_anchors)

    for i in range(0,N_anchors):
        mhe_avg[i] = np.mean(mhe[i])
        ekf_avg[i] = np.mean(ekf[i])

        mhe_stderror[i] = np.std(mhe[i])/np.sqrt(len(mhe[i]))
        ekf_stderror[i] = np.std(ekf[i])/np.sqrt(len(ekf[i]))

        mhe_success[i] = 100*mhe_in[i]/(mhe_in[i]+mhe_out[i])
        ekf_success[i] = 100*ekf_in[i]/(ekf_in[i]+ekf_out[i])

    fig, ax = plt.subplots()
    ax.plot(anchors, ekf_avg, '.-')
    ax.fill_between(anchors, ekf_avg-ekf_stderror, ekf_avg+ekf_stderror, alpha=0.3)
    ax.plot(anchors, mhe_avg, '.-')
    ax.fill_between(anchors, mhe_avg-mhe_stderror, mhe_avg+mhe_stderror, alpha=0.3)
    plt.ylim((0,1.25))
    axOut = ax.twinx()
    axOut.set_ylabel(r'Success rate (RMSE$<$1\,m) [\%]')
    plt.ylim((0,125))
    axOut.plot(anchors, ekf_success, 'x--')
    axOut.plot(anchors, mhe_success, 'x--')

    ax.grid(True)
    ax.set( xlabel=r'Number of Beacons $N$\textsubscript{UWB}', ylabel='mean RMSE [m]')
    ax.legend(["EKF", "MHE"])


def plot_noise_twr():
    """Plot noise model used for simulated TWR measurements

    The noise model used is the sum of a Gaussian (LOS component) and a
    Gamma Distribution (NLOS component). Several models with different
    heavy-tail scales are plotted.
    """

    scale = np.array([0.0, 0.5, 1.0, 1.5])
    sigma = 0.05
    lmbd = 3.5
    k = 2
    
    fx = [[] for _ in range(len(scale))]
    x_v = np.linspace(-0.25, 1.25, 1000)

    # Heavy tailed with Gamma
    for i,s in enumerate(scale):
        mu = s * 0.1

        # precompute some values
        sig2 = sigma*sigma
        gauss_pref = 1/(sigma*np.sqrt(2*np.pi))
        G = math.gamma(k)        

        for x in x_v:
            gauss = gauss_pref * np.exp(-(x-mu)*(x-mu)/(2*sig2))
            if x>0:
                gamma = np.power(lmbd,k) * np.power(x,(k-1)) * np.exp(-lmbd*x) / G
            else:
                gamma = 0

            fx[i].append(gauss/(1+s) + s*gamma/(1+s))
    
    legend = []
    fig, ax = plt.subplots()
    for i in range(len(scale)):
        ax.plot(x_v, fx[i])
        l = r'$s$\textsubscript{ht}'
        l = l + r'= {}'.format(scale[i])
        legend.append(l)

    ax.grid(True)
    ax.set( xlabel=r'$x$ [m]', ylabel='$f(x)$')
    ax.legend(legend)

def plot_noise_tdoa():
    """Plot the noise model used for simulating TdoA measurements

    The model used is the weighted average of a Gaussian and a Cauchy
    Distribution. Several ratios are plotted.
    """
    ratio = np.array([0.0, 0.5, 1.0])
    sigma = 0.3
    gamma = 0.3

    fx = [[] for _ in range(len(ratio))]
    x_v = np.linspace(-2.0, 2.0, 1000)

    sigma2 = sigma*sigma
    gamma2 = gamma*gamma
    gauss_pref = 1/(sigma*np.sqrt(2*np.pi))
    cauchy_pref = 1/(math.pi*gamma)
    for i,r in enumerate(ratio):
        for x in x_v:
            gauss = gauss_pref * np.exp(-(x*x)/(2*sigma2))
            cauchy = cauchy_pref / (1+((x*x)/gamma2))  
            f = r*cauchy + (1-r)*gauss
            fx[i].append(f)
    
    legend = []
    fig, ax = plt.subplots()
    for i in range(len(ratio)):
        ax.plot(x_v, fx[i])
        legend.append(r"$r={}$".format(ratio[i]))

    ax.grid(True)
    ax.set( xlabel=r'$x$ [m]', ylabel='$f(x)$')
    ax.legend(legend)

if __name__ == "__main__":
    plot_heavytail('tdoa_ht_noise.csv')
    plot_heavytail('twr_ht_noise.csv')
    plot_anchors('tdoa_anchors.csv')
    plot_anchors('twr_anchors.csv')
    #plot_noise_twr()
    #plot_noise_tdoa()
    plt.show()