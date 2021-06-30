"""UWBsim - Simulator for state estimation with UWB

This script starts the graphical interface for performing estimator 
simulations with UWBsim. For more information on using UWBsim please 
refer to the README file which should be distributed with the code.
"""
from PyQt5 import QtWidgets
import sys

from interface.main_window import MainWindow


app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())