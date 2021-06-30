# uwb-simulator
The uwb-simulator is a tool to compare localization algorithms based on Ultra-Wideband ranging. The simulator currently only runs on logfiles of recorded trajectories (original IMU & groundtruth), but can be used to simulate UWB ranging data if desired.

This code was used to compare an Extended Kalman Filter (EKF) and a computationally efficient Moving Horizon Estimator (MHE) for the publication "A Computationally Efficient Moving Horizon Estimator for Ultra-Wideband Localization on Small Drones" by S. Pfeiffer, C. de Wagter and G.C.H.E. de Croon. For more details you may refer to the publication [here](Link will be updated added upon publication)

## Installation

Ideally, make use of a Python virtual environment:
```bash
$ sudo apt install python3-venv
$ git clone https://github.com/SUPfeiffer/uwb-simulator.git
$ cd uwb-simulator
$ python3 -m venv venv
$ source venv/bin/activate
```

You can exit the virtual environment by typing `deactivate` at any time. Next, update `pip` and `setuptools` and install the package and its dependencies:

```bash
$ pip install --upgrade pip && pip install --upgrade setuptools
$ pip install -e .
```

We're installing the package in editable (`-e`) mode, such that we can change it without having to install again.

## Use
Start the application by executing `UWBsim/main.py`. On the left side of the interface (Data) you can select the data which the estimators will receive as input. Select the anchors you want to use and specify their positions. You can save and load anchor positions to and from .yaml files.

On the right side (Estimators), you can select and tune the estimators.

If the GUI fails to start after pulling an update, delete the previous preference file '.preferences.yaml' and try again.

### Data

This simulator currently relies on log data for IMU measurements and groundtruth. UWB data can also be used from the same log, or it can be generated as specified by the selected anchors and noise shape. 

Our own dataset is available on [4TU Research Data](https://doi.org/10.4121/14827680). Alternatively, you can also collect logs from a Crazyflie using the [Crazyflie-suite](https://github.com/Huizerd/crazyflie-suite) or create use your own source of data (This should be csv files with the same columns as the examples in the dataset on 4TU Research Data).

### Estimators

The estimators are located in `UWBsim/estimators`. Feel free to change them or implement your own estimator that has the same structure.