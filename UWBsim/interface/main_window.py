"""The main window of the UWBSimulator GUI

This file contains the code for the main window of the graphical user 
interface (GUI) of the UWBsim. In the main window, simulation parameters
can be changed and the simulation can be started.

Classes:
    MainWindow: The main window of the simulator GUI
"""

import os
import csv
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
import pyqtgraph as pg
import yaml
import math
import numpy as np

import UWBsim
from UWBsim.interface.simulation_window import SimulationWindow
from UWBsim.interface.anchor_position_window import AnchorPositionWindow
from UWBsim.simulation import SimulationParams
import UWBsim.utils.uwb_ranging as uwb



class MainWindow(QtWidgets.QWidget):
    """Main window for the UWBsim GUI

    The main window of the UWBsim GUI can be split in two parts. On the
    left hand side, the data for the simulation can be chosen and
    adjusted. This includes selection of a logfile with input data, 
    selection of what ranging data and additional measurements to use
    and the possibility to enable and position up to eight UWB anchors.
    In the case of generated UWB measurements, the noise profile can be
    tuned as well. On the right side, the estimators for comparison can
    be selected and tuned individually.
    """

    def __init__(self, *args, **kwargs):
        """Initializes and draws the main window. """

        super(MainWindow, self).__init__(*args, **kwargs)
        self.threadpool = QtCore.QThreadPool()
        self.setWindowTitle("UWB Simulator")

        # Load previous settings if it exists, use defaults otherwise
        params_reference = SimulationParams()
        if os.path.isfile(UWBsim.PREFERENCE_FILE):
            with open(UWBsim.PREFERENCE_FILE) as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        else:
            params = params_reference

        # Initialize sub-windows
        self.anchor_position_dialog = AnchorPositionWindow(
            params.ranging.anchor_positions)
        self.sim_window = SimulationWindow()
        self.sim_window.get_sim_params = self.get_sim_params


        ###############################################
        # Create and place all elements of the window #
        ###############################################
        outerLayout = QtWidgets.QVBoxLayout()
        self.setLayout(outerLayout)    

        settingsWidget = QtWidgets.QWidget()
        settingsLayout = QtWidgets.QHBoxLayout()
        settingsWidget.setLayout(settingsLayout)
        outerLayout.addWidget(settingsWidget)

        ## DATA
        self.dataBox = QtWidgets.QGroupBox('Data')
        dataLayout = QtWidgets.QGridLayout()
        self.dataBox.setLayout(dataLayout)
        settingsLayout.addWidget(self.dataBox)
        
        dataFileBox = QtWidgets.QGroupBox('Log File')
        dataFileLayout = QtWidgets.QHBoxLayout()
        dataFileBox.setLayout(dataFileLayout)
        dataLayout.addWidget(dataFileBox, 0, 0, 1, 2)

        self.file_select_lineEdit = QtWidgets.QLineEdit(params.drone.logfile)
        file_select_button = QtWidgets.QPushButton('Browse')
        file_select_button.clicked.connect(self._file_select_button_clicked)

        dataFileLayout.addWidget(self.file_select_lineEdit)
        dataFileLayout.addWidget(file_select_button)

        ## RANGING
        rangingBox = QtWidgets.QGroupBox('UWB Ranging')
        rangingLayout = QtWidgets.QVBoxLayout()
        rangingBox.setLayout(rangingLayout)
        dataLayout.addWidget(rangingBox, 1,0)

        self.uwb_log_radio = QtWidgets.QRadioButton('Use log data')
        self.uwb_log_radio.setObjectName('OriginalRanging')
        self.uwb_generate_twr_radio = QtWidgets.QRadioButton('Generate TWR')
        self.uwb_generate_tdoa_radio = QtWidgets.QRadioButton('Generate TdoA')

        if params.ranging.source == uwb.RangingSource.LOG:
            self.uwb_log_radio.setChecked(True)
        elif params.ranging.rtype == uwb.RangingType.TWR:
            self.uwb_generate_twr_radio.setChecked(True)
        elif params.ranging.rtype == uwb.RangingType.TDOA:
            self.uwb_generate_tdoa_radio.setChecked(True)
        else:
            self.uwb_log_radio.setChecked(True)
        
        self.uwb_log_radio.toggled.connect(self._ranging_toggled)
        self.uwb_generate_twr_radio.toggled.connect(self._ranging_toggled)
        self.uwb_generate_tdoa_radio.toggled.connect(self._ranging_toggled)
        
        self.ranging_interval_lb = QtWidgets.QLabel('Ranging Interval (s):')
        self.ranging_interval_lb.setDisabled(True)
        self.ranging_interval=QtWidgets.QLineEdit(str(params.ranging.interval))
        self.ranging_interval.setValidator(QtGui.QDoubleValidator())
        self.ranging_interval.setDisabled(self.uwb_log_radio.isChecked())

        rangingLayout.addWidget(self.uwb_log_radio)
        rangingLayout.addWidget(self.uwb_generate_twr_radio)
        rangingLayout.addWidget(self.uwb_generate_tdoa_radio)
        rangingLayout.addWidget(self.ranging_interval_lb)
        rangingLayout.addWidget(self.ranging_interval)

        ## NOISE
        self.noiseBox = QtWidgets.QGroupBox('Noise')
        self.noiseBox.setDisabled(self.uwb_log_radio.isChecked())
        noiseLayout = QtWidgets.QHBoxLayout()
        self.noiseBox.setLayout(noiseLayout)
        dataLayout.addWidget(self.noiseBox,1,1,2,1)
        
        noiseSettingsBox = QtWidgets.QWidget()
        noiseSettingsLayout = QtWidgets.QFormLayout()
        noiseSettingsBox.setLayout(noiseSettingsLayout)
        noiseLayout.addWidget(noiseSettingsBox)
        
        # Gaussian
        self.noise_gaussian_radio = QtWidgets.QRadioButton()
        self.noise_gaussian_radio.setChecked(
            params.ranging.source==uwb.RangingSource.GENERATE_GAUSS)
        self.noise_gaussian_radio.toggled.connect(self._noise_ht_toggled)
        
        self.noise_sigma = QtWidgets.QLineEdit(str(params.ranging.gauss_sigma))
        self.noise_sigma.setValidator(QtGui.QDoubleValidator())
        self.noise_sigma.textChanged.connect(self._noise_update)
        
        # Heavy-Tailed Cauchy
        self.noise_htCauchy_radio = QtWidgets.QRadioButton()
        self.noise_htCauchy_radio.setChecked(
            params.ranging.source==uwb.RangingSource.GENERATE_HT_CAUCHY)
        self.noise_htCauchy_radio.toggled.connect(self._noise_ht_toggled)
        
        self.noise_cauchy_ratio = QtWidgets.QLineEdit(str(params.ranging.htc_ratio))
        self.noise_cauchy_ratio.setDisabled(not self.noise_htCauchy_radio.isChecked())
        self.noise_cauchy_ratio.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 1))
        self.noise_cauchy_ratio.textChanged.connect(self._noise_update)

        self.noise_gamma = QtWidgets.QLineEdit(str(params.ranging.htc_gamma))
        self.noise_gamma.setDisabled(not self.noise_htCauchy_radio.isChecked())
        self.noise_gamma.setValidator(QtGui.QDoubleValidator())
        self.noise_gamma.textChanged.connect(self._noise_update)

        # Heavy-Tailed Gamma
        self.noise_htGamma_radio = QtWidgets.QRadioButton()
        self.noise_htGamma_radio.setChecked(
            params.ranging.source==uwb.RangingSource.GENERATE_HT_GAMMA)
        self.noise_htGamma_radio.toggled.connect(self._noise_ht_toggled)
        
        self.noise_mu = QtWidgets.QLineEdit(str(params.ranging.htg_mu))
        self.noise_mu.setDisabled(not self.noise_htGamma_radio.isChecked())
        self.noise_mu.setValidator(QtGui.QDoubleValidator())
        self.noise_mu.textChanged.connect(self._noise_update)
        
        self.noise_lambda = QtWidgets.QLineEdit(str(params.ranging.htg_lambda))
        self.noise_lambda.setDisabled(not self.noise_htGamma_radio.isChecked())
        self.noise_lambda.setValidator(QtGui.QDoubleValidator())
        self.noise_lambda.textChanged.connect(self._noise_update)
        
        self.noise_k = QtWidgets.QLineEdit(str(params.ranging.htg_k))
        self.noise_k.setDisabled(not self.noise_htGamma_radio.isChecked()) 
        self.noise_k.setValidator(QtGui.QIntValidator())
        self.noise_k.textChanged.connect(self._noise_update)
        
        self.noise_scale = QtWidgets.QLineEdit(str(params.ranging.htg_scale))
        self.noise_scale.setDisabled(not self.noise_htGamma_radio.isChecked())
        self.noise_scale.setValidator(QtGui.QDoubleValidator())
        self.noise_scale.textChanged.connect(self._noise_update)
        
        # Add parameters to layout
        #noiseSettingsLayout.addRow('Outliers (%):', self.noise_outlier_chance)
        noiseSettingsLayout.addRow('Pure Gaussian', self.noise_gaussian_radio)
        noiseSettingsLayout.addRow('Sigma:', self.noise_sigma)
        noiseSettingsLayout.addRow('HT Cauchy:', self.noise_htCauchy_radio)
        noiseSettingsLayout.addRow('Ratio (TdoA):', self.noise_cauchy_ratio)
        noiseSettingsLayout.addRow('Gamma:', self.noise_gamma)
        noiseSettingsLayout.addRow('HT Gamma:', self.noise_htGamma_radio)
        noiseSettingsLayout.addRow('mu:', self.noise_mu)
        noiseSettingsLayout.addRow('lambda:', self.noise_lambda)
        noiseSettingsLayout.addRow('k:', self.noise_k)
        noiseSettingsLayout.addRow('scale:', self.noise_scale)

        # Plot area for noise shape
        noise_plot_wg = pg.GraphicsLayoutWidget()
        noise_plot_wg.setBackground('#FAFAFA')
        noiseLayout.addWidget(noise_plot_wg)
        self.noise_plot = noise_plot_wg.addPlot()
        self.noise_plot.setTitle('Noise PDF')
        self.noise_plot.setLabels(left='f(x)', bottom='x')
        self.noise_plot.showGrid(x=True, y=True)        
        self.noise_plot.addItem(pg.InfiniteLine(pos=[0,0], pen='#AAAAAA'),
                                ignoreBounds=True)
        self.noise_plot_line = self.noise_plot.plot([], [], pen='#1F77B4')

        self._noise_update()


        ## MEASUREMENTS
        measurementBox = QtWidgets.QGroupBox('Measurements')
        measurementLayout = QtWidgets.QGridLayout()
        measurementBox.setLayout(measurementLayout)
        dataLayout.addWidget(measurementBox, 2,0)

        self.altitude_checkbox = QtWidgets.QCheckBox('Altitude')
        self.altitude_checkbox.setChecked(params.drone.altitude_enable)

        measurementLayout.addWidget(self.altitude_checkbox,1,0)

        ## ANCHORS
        anchorBox = QtWidgets.QGroupBox('Anchors')
        anchorLayout = QtWidgets.QGridLayout()
        anchorBox.setLayout(anchorLayout)
        
        anchor_enable = [False for _ in range(8)]
        for i,en in enumerate(params.ranging.anchor_enable):
            anchor_enable[i] = False or en

        self.anchor_enable_checkbox = []
        i = 0
        for row in range(2):
            for col in range(4):
                checkbox = QtWidgets.QCheckBox(
                    'ID {}\n({:.2f},{:.2f},{:.2f})'.format(
                        i, *self.anchor_position_dialog.anchor_positions[i]))
                checkbox.setChecked(anchor_enable[i])
                checkbox.setStyleSheet("QCheckBox:unchecked {color: gray}")
                self.anchor_enable_checkbox.append(checkbox)
                anchorLayout.addWidget(checkbox, row, col)
                i += 1
        
        anchor_position_button = QtWidgets.QPushButton('Anchor Positions')
        anchor_position_button.clicked.connect(self._anchor_pos_button_clicked)
        anchorLayout.addWidget(anchor_position_button, 2,0,3,4)

        dataLayout.addWidget(anchorBox, 3, 0, 4, 2)

        # ESTIMATORS
        estimatorBox = QtWidgets.QGroupBox('Estimators')
        estimatorLayout = QtWidgets.QGridLayout()
        estimatorBox.setLayout(estimatorLayout)
        settingsLayout.addWidget(estimatorBox)
        
        # MHE
        self.mheBox = QtWidgets.QGroupBox('MHE')
        self.mheBox.setCheckable(True)
        self.mheBox.setChecked(params.estimators.mhe.enable)
        mheLayout = QtWidgets.QFormLayout()
        self.mheBox.setLayout(mheLayout)
        estimatorLayout.addWidget(self.mheBox,0,0,1,1)

        self.mhe_rate = QtWidgets.QLineEdit(
            str(params.estimators.mhe.rate))
        self.mhe_rate.setValidator(QtGui.QIntValidator())

        self.mhe_Nmax = QtWidgets.QLineEdit(
            str(params.estimators.mhe.N_max))
        self.mhe_Nmax.setValidator(QtGui.QIntValidator())
        
        self.mhe_MHEIter = QtWidgets.QLineEdit(
            str(params.estimators.mhe.iterations))
        self.mhe_MHEIter.setValidator(QtGui.QIntValidator())
        
        self.mhe_mu = QtWidgets.QLineEdit(str(params.estimators.mhe.mu))
        self.mhe_mu.setValidator(QtGui.QDoubleValidator())

        self.mhe_alpha = QtWidgets.QLineEdit(
            str(params.estimators.mhe.alpha))
        self.mhe_alpha.setValidator(QtGui.QDoubleValidator())
        
        self.mhe_RANSACIter = QtWidgets.QLineEdit(
            str(params.estimators.mhe.ransac_iterations))
        self.mhe_RANSACIter.setValidator(QtGui.QIntValidator())
        
        self.mhe_RANSACFraction = QtWidgets.QLineEdit(
            str(params.estimators.mhe.ransac_fraction))
        self.mhe_RANSACFraction.setValidator(QtGui.QDoubleValidator())
        
        self.mhe_RANSACthreshold = QtWidgets.QLineEdit(
            str(params.estimators.mhe.ransac_threshold))
        self.mhe_RANSACthreshold.setValidator(QtGui.QDoubleValidator())

        mheLayout.addRow(QtWidgets.QLabel('Rate [Hz]:'), self.mhe_rate)
        mheLayout.addRow(QtWidgets.QLabel('N_max:'), self.mhe_Nmax)
        mheLayout.addRow(QtWidgets.QLabel('Iterations (MHE):'), self.mhe_MHEIter)
        mheLayout.addRow(QtWidgets.QLabel('Mu:'), self.mhe_mu)
        mheLayout.addRow(QtWidgets.QLabel('Alpha:'), self.mhe_alpha)
        mheLayout.addRow(QtWidgets.QLabel('Iterations (RANSAC):'), self.mhe_RANSACIter)
        mheLayout.addRow(QtWidgets.QLabel('RANSAC Fraction:'), self.mhe_RANSACFraction)
        mheLayout.addRow(QtWidgets.QLabel('RANSAC Threshold:'), self.mhe_RANSACthreshold)
        
        # EKF
        self.ekfBox = QtWidgets.QGroupBox('EKF')
        self.ekfBox.setCheckable(True)
        self.ekfBox.setChecked(params.estimators.ekf.enable)
        ekfLayout = QtWidgets.QFormLayout()
        self.ekfBox.setLayout(ekfLayout)
        estimatorLayout.addWidget(self.ekfBox,1,0,1,1)

        self.ekf_rate = QtWidgets.QLineEdit(str(params.estimators.ekf.rate))
        self.ekf_rate.setValidator(QtGui.QIntValidator())
        ekfLayout.addRow(QtWidgets.QLabel('Rate [Hz]:'), self.ekf_rate)

        self.ekf_outlierThreshold = QtWidgets.QLineEdit(
            str(params.estimators.ekf.outlierThreshold))
        self.ekf_outlierThreshold.setValidator(QtGui.QDoubleValidator())
        ekfLayout.addRow(QtWidgets.QLabel('Outlier Threshold:'), self.ekf_outlierThreshold)

        ## RUN BUTTON
        run_button = QtWidgets.QPushButton('Run')
        run_button.clicked.connect(self._run_button_clicked)
        estimatorLayout.addWidget(run_button, 2, 0)


    def _file_select_button_clicked(self):
        """Opens the logfile selection dialog

        Opens the dialog to navigate the filebrowser and select a logfile, then
        updates the textline that contains the location of the logfile. The 
        dialog opens in the 'data' folder. If the interaction is cancelled,
        the previous logfile is retained in the textline. 
        """

        old_logfile = self.file_select_lineEdit.text()
        logfile, _ = QtWidgets.QFileDialog.getOpenFileName(self.dataBox, 
                                                            'Open File','data')
        if logfile == "":
            self.file_select_lineEdit.setText(old_logfile)
        else:
            self.file_select_lineEdit.setText(logfile)
            self.logfile = logfile
        

    def _ranging_toggled(self):
        """Enables/Disables ranging options based on ranging mode

        Disables noise tuning and ranging interval inputs if the ranging source
        is set to log and enables them otherwise. Call after ranging source has
        been changed.
        """
        if self.uwb_log_radio.isChecked():
            self.noiseBox.setDisabled(True)
            self.ranging_interval.setDisabled(True)
            self.ranging_interval_lb.setDisabled(True)
        else:
            self.noiseBox.setDisabled(False)
            self.ranging_interval.setDisabled(False)
            self.ranging_interval_lb.setDisabled(False)
            self._noise_update()
    
    def _noise_ht_toggled(self):
        """Enables/Disables noise options based on noise type

        Enables the noise parameters needed for the chosen noise type and 
        disables all other noise parameters, then updates the noise plot. 
        Call after noise type has been changed.
        """

        if self.noise_gaussian_radio.isChecked():
            self.noise_cauchy_ratio.setDisabled(True)
            self.noise_gamma.setDisabled(True)
            self.noise_mu.setDisabled(True)
            self.noise_lambda.setDisabled(True)
            self.noise_k.setDisabled(True)
            self.noise_scale.setDisabled(True)

        elif self.noise_htCauchy_radio.isChecked():
            self.noise_cauchy_ratio.setDisabled(False)
            self.noise_gamma.setDisabled(False)
            self.noise_mu.setDisabled(True)
            self.noise_lambda.setDisabled(True)
            self.noise_k.setDisabled(True)
            self.noise_scale.setDisabled(True)
            
        elif self.noise_htGamma_radio.isChecked():
            self.noise_cauchy_ratio.setDisabled(True)
            self.noise_gamma.setDisabled(True)
            self.noise_mu.setDisabled(False)
            self.noise_lambda.setDisabled(False)
            self.noise_k.setDisabled(False)
            self.noise_scale.setDisabled(False)
            
        self._noise_update()

    def _noise_update(self):
        """Updates plot of the noise PDF

        Updates the probability density function (PDF) of the noise displayed
        in the noise plot based on the current noise parameters. Call after
        noise parameters have been changed.
        """

        sigma = float(self.noise_sigma.text())
        if sigma==0:
            return
        
        fx = []
        x = []
        
        if self.noise_gaussian_radio.isChecked():
            # pure gaussian
            sigma2 = sigma*sigma
            gauss_pref = 1/(sigma*np.sqrt(2*np.pi))
            for i in np.linspace(-6*sigma, 6*sigma, 100):
                f = gauss_pref * np.exp(-i*i/(2*sigma2))
                x.append(i)
                fx.append(f)

        elif self.noise_htCauchy_radio.isChecked():
            # Heavy tailed with Cauchy
            sigma = float(self.noise_sigma.text())
            gamma = float(self.noise_gamma.text())
            sigma2 = sigma*sigma
            gamma2 = gamma*gamma
            
            if sigma==0 or gamma==0:
                return

            if self.uwb_generate_twr_radio.isChecked():
                alpha = (2*math.pi*gamma) / (math.sqrt(2*math.pi*sigma2) + math.pi*gamma)
                gauss_pref = (2-alpha)/(sigma*np.sqrt(2*np.pi))
                cauchy_pref = alpha/(math.pi*gamma)

                for i in np.linspace(-6*sigma, 0, 50):
                    f = gauss_pref * np.exp(-i*i/(2*sigma2))
                    x.append(i)
                    fx.append(f)

                gamma2 = gamma*gamma
                for i in np.linspace(0,12*sigma, 100):
                    f = cauchy_pref / (1+(i*i/gamma2))  
                    x.append(i)
                    fx.append(f)

            elif self.uwb_generate_tdoa_radio.isChecked():
                ratio = float(self.noise_cauchy_ratio.text())
                gauss_pref = 1/(sigma*np.sqrt(2*np.pi))
                cauchy_pref = 1/(math.pi*gamma)
                for i in np.linspace(-6*sigma, 6*sigma, 200):
                    gauss = gauss_pref * np.exp(-(i)*(i)/(2*sigma2))
                    cauchy = cauchy_pref / (1+((i*i)/gamma2))  
                    f = ratio*cauchy + (1-ratio)*gauss
                    x.append(i)
                    fx.append(f)

        elif self.noise_htGamma_radio.isChecked():
            # Heavy tailed with Gamma
            mu = float(self.noise_mu.text())
            sigma = float(self.noise_sigma.text())
            lmbd = float(self.noise_lambda.text())
            k = int(self.noise_k.text())
            alpha = float(self.noise_scale.text())

            sig2 = sigma*sigma
            gauss_pref = 1/(sigma*np.sqrt(2*np.pi))
            G = math.gamma(k)        

            for i in np.linspace(-6*sigma, 12*sigma, 150):
                gauss = gauss_pref * np.exp(-(i-mu)*(i-mu)/(2*sig2))
                if i>0:
                    gamma = np.power(lmbd,k) * np.power(i,(k-1)) * np.exp(-lmbd*i) / G
                else:
                    gamma = 0

                x.append(i)
                fx.append(gauss/(1+alpha) + alpha*gamma/(1+alpha))
        
        

        self.noise_plot_line.setData(x, fx)


    def _anchor_pos_button_clicked(self):
        """Opens dialog to edit, save and load anchor positions """
        self.anchor_position_dialog.exec_()
        for i, anchor in enumerate(self.anchor_enable_checkbox):
            anchor.setText('ID {}\n({:.2f},{:.2f},{:.2f})'.format(i, 
                                            *self.anchor_position_dialog.anchor_positions[i]))


    def _run_button_clicked(self):
        """Starts estimator simulation in separate window """
        self.sim_window.run_button_clicked()

        # Show/Bring window to front
        if self.sim_window.isVisible():
            self.sim_window.activateWindow()
        else:
            self.sim_window.show()

    def get_sim_params(self):
        """Collects and returns simulation parameters from the main window

        Crates a SimulationParams parameter structure with all the parameters
        that were entered into the main window. 

        Returns:
            Instance of SimulationParams containing all the parameter choices
            made.
        """

        params = SimulationParams()
        # logfile and name
        params.drone.logfile = self.file_select_lineEdit.text()
        (_,params.name) = os.path.split(params.drone.logfile)

        # ESTIMATORS
        # MHE
        params.estimators.mhe.enable = self.mheBox.isChecked()
        params.estimators.mhe.rate = int(self.mhe_rate.text())
        params.estimators.mhe.N_max = int(self.mhe_Nmax.text())
        params.estimators.mhe.iterations = int(self.mhe_MHEIter.text())
        params.estimators.mhe.mu = float(self.mhe_mu.text())
        params.estimators.mhe.alpha = float(self.mhe_alpha.text())
        params.estimators.mhe.ransac_iterations = int(self.mhe_RANSACIter.text())
        params.estimators.mhe.ransac_fraction = float(self.mhe_RANSACFraction.text())
        params.estimators.mhe.ransac_threshold = float(self.mhe_RANSACthreshold.text())
        
        # EKF
        params.estimators.ekf.enable = self.ekfBox.isChecked()
        params.estimators.ekf.rate = int(self.ekf_rate.text())
        params.estimators.ekf.outlierThreshold = float(self.ekf_outlierThreshold.text())

        # MEASUREMENTS
        # altitude
        params.drone.altitude_enable = self.altitude_checkbox.isChecked()
        # uwb anchors
        params.ranging.anchor_positions = self.anchor_position_dialog.anchor_positions.tolist()
        params.ranging.anchor_enable = []
        for anchor in self.anchor_enable_checkbox:
            params.ranging.anchor_enable.append(anchor.isChecked())
        # ranging source and type
        if self.uwb_log_radio.isChecked():
            params.ranging.source = uwb.RangingSource.LOG
            with open(params.drone.logfile, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                header = next(csv_reader)
                for h in header:
                    if 'twr' in h:
                        params.ranging.rtype = uwb.RangingType.TWR
                        break
                    elif 'tdoa' in h:
                        params.ranging.rtype = uwb.RangingType.TDOA
                        break
                    else:
                        params.ranging.rtype = uwb.RangingType.NONE
        else:
            params.ranging.interval = float(self.ranging_interval.text())
            # source
            if self.noise_gaussian_radio.isChecked():
                params.ranging.source = uwb.RangingSource.GENERATE_GAUSS
            elif self.noise_htCauchy_radio.isChecked():
                params.ranging.source = uwb.RangingSource.GENERATE_HT_CAUCHY
            elif self.noise_htGamma_radio.isChecked():
                params.ranging.source = uwb.RangingSource.GENERATE_HT_GAMMA
            else:
                params.ranging.source = uwb.RangingSource.NONE
            # type
            if self.uwb_generate_twr_radio.isChecked():
                params.ranging.rtype = uwb.RangingType.TWR
            elif self.uwb_generate_tdoa_radio.isChecked():
                params.ranging.rtype = uwb.RangingType.TDOA
            else:
                params.ranging.rtype = uwb.RangingType.NONE
        
        # NOISE
        params.ranging.gauss_sigma = float(self.noise_sigma.text())
        params.ranging.htc_gamma = float(self.noise_gamma.text())
        params.ranging.htc_ratio = float(self.noise_cauchy_ratio.text())
        params.ranging.htg_mu = float(self.noise_mu.text())
        params.ranging.htg_lambda = float(self.noise_lambda.text())
        params.ranging.htg_k = int(self.noise_k.text())
        params.ranging.htg_scale = float(self.noise_scale.text())

        return params


    def closeEvent(self, event):
        """On close, save previous parameters for next run
        
        This function adds additional action to be executed when the main
        window is closed, namely dumping the current parameter choices to 
        a preference file, which is specifying the initial parameters when the
        program is started the next time.
        """

        anchor_enable = []
        for anchor in self.anchor_enable_checkbox:
            anchor_enable.append(anchor.isChecked())
        
        settings = self.get_sim_params()
        settings.ranging.rtype = int(settings.ranging.rtype)
        settings.ranging.source = int(settings.ranging.source)
        with open(UWBsim.PREFERENCE_FILE, 'w') as f:
            yaml.dump(settings, f)
        event.accept() 