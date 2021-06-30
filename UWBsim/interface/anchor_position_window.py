"""Dialog Window to specify anchor positions to the simulation

The AnchorPositionWindow allows entering anchor positions manually as
well as saving and loading with yaml files.

Classes:
    AnchorPositionWindow: Dialog to specify anchor positions
"""

from PyQt5 import QtWidgets
from PyQt5 import QtGui
import numpy as np
import yaml
import UWBsim


class AnchorPositionWindow(QtWidgets.QDialog):
    """QDialog that allows entering anchors positions for the simulation

    The AnchorPositionWindow is used to let the user enter the anchor
    positions manually and save them to a yaml file. Previously saved
    yaml files can also be loaded. The calling program can access the 
    anchor positions through the anchor_position variable. The window
    allows setting positions for 8 anchors.
    """

    def __init__(self, anchor_positions, *args, **kwargs):
        """Initializes AnchorPositionWindow

        Initializes the window with a provided set of anchor positions
        """

        super(AnchorPositionWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('Anchor Positions')

        # Create the variable that will hold the anchor positions
        self.anchor_positions = np.zeros((8,3))
        for i,p in enumerate(anchor_positions):
            self.anchor_positions[i] = p
        
        # Draw the window layout
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.anchorBox = []
        self.anchorLineEdits = []
        i = 0
        for row in range(2):
            for col in range(4):
                self.anchorBox.append(QtWidgets.QGroupBox('ID{}'.format(i)))

                anchorLayout = QtWidgets.QFormLayout()
                self.anchorBox[i].setLayout(anchorLayout)

                xline = QtWidgets.QLineEdit(str(self.anchor_positions[i][0]))
                xline.setValidator(QtGui.QDoubleValidator())
                yline = QtWidgets.QLineEdit(str(self.anchor_positions[i][1]))
                yline.setValidator(QtGui.QDoubleValidator())
                zline = QtWidgets.QLineEdit(str(self.anchor_positions[i][2]))
                zline.setValidator(QtGui.QDoubleValidator())
                self.anchorLineEdits.append([xline, yline, zline])
                anchorLayout.addRow('x', xline)
                anchorLayout.addRow('y', yline)
                anchorLayout.addRow('z', zline)

                layout.addWidget(self.anchorBox[i], row, col)
                i = i+1
        
        self.ok_button = QtWidgets.QPushButton('OK')
        self.ok_button.clicked.connect(self._ok_button_clicked)
        layout.addWidget(self.ok_button, 2,3)

        self.save_button = QtWidgets.QPushButton('Save as file')
        self.save_button.clicked.connect(self._save_button_clicked)
        layout.addWidget(self.save_button, 2,1)

        self.load_button = QtWidgets.QPushButton('Load')
        self.load_button.clicked.connect(self._load_button_clicked)
        layout.addWidget(self.load_button, 2,2)


    def _load_anchor_positions(self, anchor_file):
        """Load anchor positions from a yaml file

        Open a yaml file and set the anchor_position variable with the 
        values contained in it. Then update the values displayed by the
        dialog window.

        Args:
            anchor_file: File containing the anchor positions
        """

        with open(anchor_file) as f:
            positions = yaml.safe_load(f)     
            for key,pos in positions.items():
                i = int(key)
                self.anchor_positions[i][0] = pos['x']
                self.anchor_positions[i][1] = pos['y']
                self.anchor_positions[i][2] = pos['z']

        for i in range(8):
            self.anchorLineEdits[i][0].setText(str(self.anchor_positions[i][0]))
            self.anchorLineEdits[i][1].setText(str(self.anchor_positions[i][1]))
            self.anchorLineEdits[i][2].setText(str(self.anchor_positions[i][2]))


    def _save_button_clicked(self):
        """Save the anchor positions to a yaml file
        
        Opens a file dialog to specify a safe file, then dumps the current
        values entered in the interface to the file.
        """

        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"Save File",UWBsim.BASE_DIR,"All Files (*);;YAML files (*.yaml)")
        
        yaml_dump = {}
        for i in range(len(self.anchor_positions)):
            key = str(i)
            yaml_dump[key] = {}
            yaml_dump[key]['x'] = str(self.anchorLineEdits[i][0].text())
            yaml_dump[key]['y'] = str(self.anchorLineEdits[i][1].text())
            yaml_dump[key]['z'] = str(self.anchorLineEdits[i][2].text())

        if not fileName.endswith('.yaml'):
            fileName = fileName + '.yaml'
        
        with open(fileName, 'w') as f:
            yaml.safe_dump(yaml_dump, f)


    def _load_button_clicked(self):
        """Load anchor positions from a yaml file

        Opens a file dialog for user to select a file to load from, then
        loads anchor positions from that file.
        """

        anchor_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load', UWBsim.BASE_DIR,'*.yaml')
        if anchor_file:
            self.anchor_file = anchor_file
            self._load_anchor_positions(anchor_file)


    def _ok_button_clicked(self):
        """Update anchor_positions and close the dialog

        Updates the anchor_position variable with the values entered into
        the dialog, then closes the dialog window (without destroying the
        instance of the window, so the variable can still be accessed 
        from the calling program)
        """
        for i in range(8):
            self.anchor_positions[i][0] = self.anchorLineEdits[i][0].text()
            self.anchor_positions[i][1] = self.anchorLineEdits[i][1].text()
            self.anchor_positions[i][2] = self.anchorLineEdits[i][2].text()
        
        self.accept()