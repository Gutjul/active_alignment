# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:36:06 2023

@author: s176369
"""
import numpy as np
import time
import pickle
from PyQt6 import QtGui
from PyQt6 import QtCore
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QGridLayout,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
)
from PyQt6.QtGui import QPixmap, QKeyEvent, QDoubleValidator
from PySide6.QtDataVisualization import Q3DSurface
from PySide6.QtGui import QBrush, QIcon, QLinearGradient, QPainter
from active_alignment_setup import Active_Alignment_Setup
from random import choice
import sys
from time import sleep
from decimal import Decimal
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import TimedAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from bsc203 import BSC203
#from santec_gui import Santec_GUI
import pyvisa
import nidaqmx
# rm=pyvisa.ResourceManager()
# listing=rm.list_resources() #creat a list of all detected connections
# system=nidaqmx.system.System.local()
class MainWindow(QMainWindow):
    def __init__(self):
        """
        Graphical user interface (GUI) based on PyQt6 for active alignment setup
        The first initialization section is rather long since all elements in
        the window have to be added.
        
        """
        super().__init__()
        self.setWindowTitle("Active alignment setup")

        self.setup = Active_Alignment_Setup(simulate = False, zero = False,
                            piezo_ID = '71345664', stepper_ID = '70391704') # Intiate setup object
        layout_controller = QGridLayout()
        layoutbig = QGridLayout()
        self.x_box = QDoubleSpinBox()
        x_label = QLabel()
        x_label.setText("X (microns):")
        self.y_box = QDoubleSpinBox()
        y_label = QLabel()
        y_label.setText("Y (microns):")
        self.z_box = QDoubleSpinBox()
        z_label = QLabel()
        z_label.setText("Z (microns):")
        self.home_x_piezo_button = QPushButton("Zero")
        self.home_x_piezo_button.clicked.connect(lambda: self.zero_piezo_channel("x"))
        self.home_y_piezo_button = QPushButton("Zero")
        self.home_y_piezo_button.clicked.connect(lambda: self.zero_piezo_channel("y"))
        self.home_z_piezo_button = QPushButton("Zero")
        self.home_z_piezo_button.clicked.connect(lambda: self.zero_piezo_channel("z"))
        self.boxes = [self.x_box, self.y_box, self.z_box]
        self.signal = QLineEdit()
        font = self.signal.font()
        font.setPointSize(25)
        self.signal.setFont(font)
        self.signal.setReadOnly(True)
        self.signal.setMaximumWidth(200)
        self.signal_label = QLabel("Coupling power [V]:")
        self.signal_button = QPushButton("Read signal")
        self.signal_button.setCheckable(True)
        self.signal_button.toggled.connect(self.signal_button_is_toggled)

        for box in self.boxes:
            box.setMinimum(0), box.setMaximum(30)
            box.setDecimals(4)
            if box == self.z_box:
                box.setValue(30)
            else:
                box.setValue(15)
            box.editingFinished.connect(self.pos_value_changed)
            box.setSingleStep(0.5)

        self.x_stepper_box = QDoubleSpinBox()

        self.y_stepper_box = QDoubleSpinBox()

        self.z_stepper_box = QDoubleSpinBox()
        
        for i, box in enumerate([self.x_stepper_box, self.y_stepper_box, self.z_stepper_box]):
            if box == self.z_stepper_box:
                
                box.setMinimum(0), box.setMaximum(6)
            else:
                box.setMinimum(0), box.setMaximum(4)
            box.setDecimals(5)
            box.setValue(self.setup.position[i + 3])
            box.editingFinished.connect(self.pos_value_changed)
            box.setSingleStep(0.1)
        self.keyboard_control_button = QPushButton("Keyboard control (Q, W, E, A, S, D)")
        self.keyboard_control_button.setCheckable(True)
        self.disable_x_stepper_button = QPushButton("Disable")
        self.disable_x_stepper_button.setCheckable(True)
        self.disable_x_stepper_button.toggled.connect(lambda: self.disable_channel(self.disable_x_stepper_button, 1))
        self.disable_y_stepper_button = QPushButton("Disable")
        self.disable_y_stepper_button.setCheckable(True)
        self.disable_y_stepper_button.toggled.connect(lambda: self.disable_channel(self.disable_y_stepper_button, 2))
        self.disable_z_stepper_button = QPushButton("Disable")
        self.disable_z_stepper_button.setCheckable(True)
        self.disable_z_stepper_button.toggled.connect(lambda: self.disable_channel(self.disable_z_stepper_button, 3))
        self.home_x_stepper_button = QPushButton("Home")
        self.home_x_stepper_button.clicked.connect(lambda: self.home_stepper_channel(1))
        self.home_y_stepper_button = QPushButton("Home")
        self.home_y_stepper_button.clicked.connect(lambda: self.home_stepper_channel(2))
        self.home_z_stepper_button = QPushButton("Home")
        self.home_z_stepper_button.clicked.connect(lambda: self.home_stepper_channel(3))
        self.home_all_button = QPushButton("Home all")
        self.home_all_button.clicked.connect(self.home_all_controllers)
        self.home_status_label = QLabel("")
        
        self.keyboard_control_message = QLabel("   ")
        self.keyboard_control_message.setStyleSheet('color: red')
        
        layout_controller.addWidget(QLabel("Piezo coordinates:"), 0, 0)
        layout_controller.addWidget(QLabel("X (microns):"), 1, 0)
        layout_controller.addWidget(self.x_box, 1, 1)
        layout_controller.addWidget(self.home_x_piezo_button, 1, 2)
        layout_controller.addWidget(QLabel("Y (microns):"), 2, 0)
        layout_controller.addWidget(self.y_box, 2, 1)
        layout_controller.addWidget(self.home_y_piezo_button, 2, 2)
        layout_controller.addWidget(QLabel("Z (microns):"), 3, 0)
        layout_controller.addWidget(self.z_box, 3, 1)
        layout_controller.addWidget(self.home_z_piezo_button, 3, 2)
        layout_controller.addWidget(QLabel("Stepper coordinates:"), 4, 0)
        layout_controller.addWidget(QLabel("X (mm):"), 5, 0)
        layout_controller.addWidget(self.x_stepper_box, 5, 1)
        layout_controller.addWidget(self.disable_x_stepper_button, 5, 2)
        layout_controller.addWidget(self.home_x_stepper_button, 5, 3)
        layout_controller.addWidget(QLabel("Y (mm):"), 6, 0)
        layout_controller.addWidget(self.y_stepper_box, 6, 1)
        layout_controller.addWidget(self.disable_y_stepper_button, 6, 2)
        layout_controller.addWidget(self.home_y_stepper_button, 6, 3)
        layout_controller.addWidget(QLabel("Yaw (degrees):"), 7, 0)
        layout_controller.addWidget(self.z_stepper_box, 7, 1)
        layout_controller.addWidget(self.disable_z_stepper_button, 7, 2)
        layout_controller.addWidget(self.home_z_stepper_button, 7, 3)
        layout_controller.addWidget(self.home_all_button, 8, 0)
        layout_controller.addWidget(self.home_status_label, 8, 1)
        layout_controller.addWidget(self.keyboard_control_button, 9, 0, 1, 2)
        layout_controller.addWidget(self.keyboard_control_message, 9, 2, 1, 2)
        layout_controller.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        
        
        # Add powermeter section
        powermeter_column = 4
        self.powcombobox = QComboBox()
        for name in self.setup.power_readers:
            self.powcombobox.addItem(name)
        
        self.powcombobox.setCurrentIndex(0)
        self.powcombobox.currentIndexChanged.connect(self.powselectionchange)
        self.refresh_pow_button = QPushButton('Refresh')
        self.refresh_pow_button.clicked.connect(self.refresh_pow)
        layout_controller.addWidget(QLabel("Power meter section"), 0, powermeter_column)
        layout_controller.addWidget(QLabel("Photodetector:"), 1, powermeter_column)
        layout_controller.addWidget(self.powcombobox, 1, powermeter_column + 1)
        layout_controller.addWidget(self.refresh_pow_button, 1, powermeter_column + 2)

        layout_controller.addWidget(self.signal, 2, powermeter_column, 2, 2)
        layout_controller.addWidget(self.signal_button, 2, powermeter_column + 2)
        
        layoutbig.addLayout(layout_controller, 0, 0)
        
        
        # add raster scanning section 
        # First for stepper motor
        raster_column = 7
        raster_row = 0
        layout_raster = QGridLayout()
        layout_controller.addWidget(QLabel("Stepper raster section"), raster_row, raster_column)
        stepper_raster_button = QPushButton("Run stepper raster")
        stepper_raster_button.clicked.connect(self.run_stepper_raster)
        
        # Stepper scan step size (sss)
        self.stepper_step_size_box = QLineEdit("0.01")
        self.stepper_step_size_box.textChanged.connect(self.sss_value_changed)
        self.stepper_step_size_box.setValidator(QDoubleValidator(0, 1, 10))
        self.stepper_step_size_box.setMaximumWidth(75)
        layout_controller.addWidget(QLabel("Step size [mm]"), 1, raster_column)
        layout_controller.addWidget(self.stepper_step_size_box, raster_row + 1, raster_column + 1, 1, 3)
        
        
        #layout_raster.addLayout(layout_sss, 1, 0, 1, 3)
        
        # Stepper scan step size width (sssw)
        self.stepper_step_scan_width_x = QLineEdit("0.1")
        self.stepper_step_scan_width_x.setMaximumWidth(75)
        self.stepper_step_scan_width_x.setValidator(QDoubleValidator(0, 1, 10))  # Setting bounds for step size edit boxes

        self.stepper_step_scan_width_y = QLineEdit("0.1")
        self.stepper_step_scan_width_y.setMaximumWidth(75)
        self.stepper_step_scan_width_y.setValidator(QDoubleValidator(0, 1, 10))  # Setting bounds for step size edit boxes
    
        layout_controller.addWidget( QLabel("Scan width (x, y) [mm]"), raster_row + 2, raster_column, 1, 1)
        layout_controller.addWidget(self.stepper_step_scan_width_x, raster_row + 2, raster_column + 1, 1, 3)
        layout_controller.addWidget(self.stepper_step_scan_width_y, raster_row + 2, raster_column + 4, 1, 3)
        
        
        #layout_controller.addLayout(layout_sssw, 2, 0, 1, 3)
        
        
        
        layout_controller.addWidget(stepper_raster_button, raster_row + 3, raster_column + 1, 1, 6)
        
        
        # add piezo raster section
        
        layout_controller.addWidget(QLabel("Piezo raster section"), raster_row + 4, raster_column)
        piezo_raster_button = QPushButton("Run piezo raster")
        piezo_raster_button.clicked.connect(self.run_piezo_raster)
        
        # Stepper scan step size (sss)
        self.piezo_step_size_box = QLineEdit("0.2")
        # self.piezo_step_size_box.textChanged.connect(self.piezo_sss_value_changed)
        self.piezo_step_size_box.setValidator(QDoubleValidator(0, 1, 10))
        self.piezo_step_size_box.setMaximumWidth(75)
        layout_controller.addWidget(QLabel("Step size [V]"), raster_row + 5, raster_column)
        layout_controller.addWidget(self.piezo_step_size_box, raster_row + 5, raster_column + 1, 1, 3)

        # Stepper scan step size width (sssw)

        self.piezo_step_scan_width_x = QLineEdit("30")
        self.piezo_step_scan_width_x.setMaximumWidth(75)
        self.piezo_step_scan_width_x.setValidator(QDoubleValidator(0, 1, 10))  # Setting bounds for step size edit boxes

        self.piezo_step_scan_width_y = QLineEdit("30")
        self.piezo_step_scan_width_y.setMaximumWidth(75)
        self.piezo_step_scan_width_y.setValidator(QDoubleValidator(0, 1, 10))  # Setting bounds for step size edit boxes
        
        layout_controller.addWidget(QLabel("Scan width (x, y) [microns]"), raster_row + 6, raster_column)
        layout_controller.addWidget(self.piezo_step_scan_width_x, raster_row + 6, raster_column + 1, 1, 3)
        layout_controller.addWidget(self.piezo_step_scan_width_y, raster_row + 6, raster_column + 4, 1, 3)

        
        layout_controller.addWidget(QLabel("Axes for piezo raster:"), raster_row + 7, raster_column, 1, 1)
        
        layout_controller.addWidget(QLabel("x:"), raster_row + 7, raster_column + 1, 1, 1)
        
        self.piezo_x_checkbox = QCheckBox()
        self.piezo_x_checkbox.setChecked(True)
        self.piezo_y_checkbox = QCheckBox()
        self.piezo_y_checkbox.setChecked(True)
        self.piezo_z_checkbox = QCheckBox()
        layout_controller.addWidget(self.piezo_x_checkbox, raster_row + 7, raster_column + 2, 1, 1)
        layout_controller.addWidget(QLabel("y:"), raster_row + 7, raster_column + 3, 1, 1)
        
        layout_controller.addWidget(self.piezo_y_checkbox, raster_row + 7, raster_column + 4, 1, 1)
        
        layout_controller.addWidget(QLabel("z:"), raster_row + 7, raster_column + 5, 1, 1)
        
        layout_controller.addWidget(self.piezo_z_checkbox, raster_row + 7, raster_column + 6, 1, 1)
        
        layout_controller.addWidget(piezo_raster_button, raster_row + 8, raster_column + 1, 1, 6)
            
        # layoutbig.addLayout(layout_raster, 0, 2, 3, 1)
        
        # Now the optimization part
        
        optim_column = 7
        optim_row = 9
        
        self.optim_res = QLineEdit()
        self.optim_res.setReadOnly(True)

        layout_controller.addWidget(QLabel("Optimization results:"), optim_row, optim_column)
        
        layout_controller.addWidget(self.optim_res, optim_row, optim_column + 1, 1, 6)
        
        layout_init5 = QHBoxLayout()
        self.optimize_button_stepper = QPushButton("Quick optimize")
        self.optimize_button_stepper.clicked.connect(self.quick_optimize)
        
        
        layout_controller.addWidget(self.optimize_button_stepper, optim_row + 1, optim_column + 1, 1, 6)

        self.piezo_optimize_button = QPushButton("Piezo optimize (xy)")
        self.piezo_optimize_button.clicked.connect(lambda: self.optimize(
            axes = [0, 1], step_sizes = [0.1, 0.1], method = "Hill climb"))
        layout_controller.addWidget(self.piezo_optimize_button, optim_row + 2, optim_column + 1, 1, 3)
        self.optimize_with_z_button = QPushButton("Piezo optimize (xyz)")
        self.optimize_with_z_button.clicked.connect(lambda: self.optimize(
            axes = [0, 1, 2], step_sizes = [0.1, 0.1, 1.0], method = "Pattern search"))
        layout_controller.addWidget(self.optimize_with_z_button, optim_row + 2, optim_column + 4, 1, 3)
        self.save_button = QPushButton("Save")
        
        self.save_button.clicked.connect(self.saveFunc)
        layout_controller.addWidget(self.save_button, optim_row + 3, optim_column + 1, 1, 6)
    
    
    
    
        self.optimize_method = QComboBox()
        self.optimize_method.addItem("Pattern search")
        self.optimize_method.addItem("Hill climb")
        self.optimize_method.addItem("Dichotomy")
        
        self.axis_check_boxes = [QCheckBox(), QCheckBox(), QCheckBox(), QCheckBox(), QCheckBox(), QCheckBox()]
        self.step_size_boxes = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.optimize)
        
        optimize_layout = QGridLayout()
        optimize_layout.addWidget(QLabel("Customizable optimization section"), 0, 0)
        optimize_layout.addWidget(QLabel("Method:"), 1, 0)
        optimize_layout.addWidget(self.optimize_method, 1, 1)
        optimize_layout.addWidget(self.optimize_button, 1, 2)
        
        optimize_layout.addWidget(QLabel("Axis:"), 2, 0)
        optimize_layout.addWidget(QLabel("Enable:"), 3, 0)
        optimize_layout.addWidget(QLabel("Step size:"), 4, 0)
        bounds = [[0, 30], [0, 30], [0, 30], [0, 4], [0, 4], [0, 6]]
        initial_values = [0.1, 0.1, 1.0, 0.0005, 0.0005, 0.01]
        for i, label in enumerate(["x piezo", "y piezo", "z piezo", "x stepper", "y stepper", "yaw stepper"]):
            optimize_layout.addWidget(QLabel(label), 3, i + 1)
            optimize_layout.addWidget(self.axis_check_boxes[i], 4, i + 1)
            optimize_layout.addWidget(self.step_size_boxes[i], 5, i + 1)
            self.step_size_boxes[i].setText(str(initial_values[i]))
            self.step_size_boxes[i].setValidator(
                QDoubleValidator(bounds[i][0], bounds[i][1], 10))  # Setting bounds for step size edit boxes
            

        # layout_raster.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Add the error message window
        #error_row = 12
        #self.error_message_board = QTextEdit()
        #layout_controller.addWidget(self.error_message_board, error_row, 0, 1, 4)
        
        
        self.signal_timer = QTimer(self)
        self.signal_timer.timeout.connect(self.update_power_reading)

        # add plot !!!!
        keyboard_layout = QVBoxLayout()
        self.keyboard_label = QLabel()
        
        self.keyboard_map = QPixmap('keyboard_description.png')
        # self.keyboard_label.setPixmap(self.keyboard_map)
        self.keyboard_control_timer = QTimer(self)

        # keyboard_layout.addWidget(self.keyboard_control_button)
        
        #keyboard_layout.addWidget(self.keyboard_control_message)
        self.keyboard_control_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        keyboard_layout.addWidget(self.keyboard_label)
        keyboard_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
 
        self.raster_plot = QPixmap()
        self.raster_label = QLabel()
        self.raster_label.setPixmap(self.raster_plot)
        layoutbotter = QHBoxLayout()
        layoutbotter.addLayout(keyboard_layout)
        layoutbotter.addWidget(self.raster_label)
        self.plot_raster(np.array([]), np.array([]), np.array([[], []]))
        self.save_button.setEnabled(False)
        layoutbigger = QVBoxLayout()
        layoutbigger.addLayout(layoutbig)
        
        
        layoutbigger.addLayout(layoutbotter)
        widget = QWidget()
        widget.setLayout(layoutbigger)
        self.setCentralWidget(widget)
        
        
        
    def pos_value_changed(self):
        """
        Function that is initiated whenever a value in one of the position
        boxes in the window is changed
        """
        self.setup.set_position([self.x_box.value(), self.y_box.value(), self.z_box.value(), 
                                 self.x_stepper_box.value(), self.y_stepper_box.value(), self.z_stepper_box.value()])
    # def mousePressEvent(self, e):
    #     self.setup.set_position([self.x_box.value(), self.y_box.value(), self.z_box.value(), 
    #                              self.x_stepper_box.value(), self.y_stepper_box.value(), self.z_stepper_box.value()])

    def sss_value_changed(self, val):
        """
        When step size value is changed in window, the jog step size of the
        stepper controller is adjusted to the chosen value
        """
        for i in range(2):
            self.setup.stepper.jog_step_size[i] = float(val)
            self.setup.stepper.jog_step_size_int[i] = int(self.setup.stepper.jog_step_size[i] * self.setup.stepper.calibration_number[i])
    # def piezo_sss_value_changed(self, val):
    #     """
    #     When step size value is changed in window, the jog step size of the
    #     stepper controller is adjusted to the chosen value
    #     """
    #     # for i in range(2):
    #     #     self.setup.stepper.jog_step_size[i] = float(val)
    #     #     self.setup.stepper.jog_step_size_int[i] = int(self.setup.jog_step_size[i] * self.setup.stepper.calibration_number[i])
    
    def update_power_reading(self):
        """
        Updates the power reading in the signal box to monitor the coupling power
        """
        self.signal.setText('%.4E' % Decimal(self.setup.read_input()))
    
    def signal_button_is_toggled(self, checked):
        """
        Initiated when "Read signal" button is pressed.
        """
        if checked == True:
            self.signal_timer.start(10)  # Update every 10 milliseconds (0.01 second)
        else:
            self.signal_timer.stop()

    def run_stepper_raster(self):
        """
        Runs a raster scan with the stepper motor controller using the values from the 
        corresponding boxes (width and step size)
        """
        X, Y, Z = self.setup.run_stepper_raster(width = [float(self.stepper_step_scan_width_x.text()),
                                                         float(self.stepper_step_scan_width_y.text())], 
                                                step_size = float(self.stepper_step_size_box.text()))
        result = self.setup.stepper.get_positions()
        # Loop below sets the found values in the position boxes of the window.
        for i, box in enumerate([self.x_stepper_box, self.y_stepper_box, self.z_stepper_box]):
            box.setValue(result[i])
        self.plot_raster(X, Y, Z.T)
        
        
    def run_piezo_raster(self):
        """
        Runs a raster scan with the stepper motor controller using the values from the 
        corresponding boxes (width and step size)
        """
        axes = []
        if self.piezo_x_checkbox.isChecked():
            axes.append(0)
        if self.piezo_y_checkbox.isChecked():
            axes.append(1)
        if self.piezo_z_checkbox.isChecked():
            axes.append(2)
        if len(axes) != 2:
            raise ValueError("Two axes can be scanned at a time")
            
        _, _, _, _, Z, X, Y = self.setup.run_piezo_raster(width = [float(self.piezo_step_scan_width_x.text()),
                                                         float(self.piezo_step_scan_width_y.text())], 
                                                step_size = float(self.piezo_step_size_box.text()),
                                                axes = axes)
        
        #result = self.setup.stepper.get_positions()
        # Loop below sets the found values in the position boxes of the window.
        # for i, box in enumerate([self.x_stepper_box, self.y_stepper_box, self.z_stepper_box]):
        #     box.setValue(result[i])
        self.plot_raster(X, Y, Z, axes)
        

    def plot_raster(self, X, Y, Z, axes = [0, 1]):
    
        """
        Simply plots the raster results in a 3D surface.
        First saves the plot as an image and then imports it to the window.
        """
        ax_labels = ["X", "Y", "Z"]
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = (8, 8))
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        # fig.set_figheight(8)
        # fig.set_figwidth(8)
        ax.set_xlabel("\n" + ax_labels[axes[0]] + " position [microns]", fontsize=14, rotation=0, linespacing = 2.7)
        ax.set_ylabel("\n" + ax_labels[axes[1]] + " position [microns]", fontsize=14, rotation=0, linespacing = 2.7)
        ax.set_zlabel("\nPhotodetector voltage [V]", fontsize=14, rotation=0, linespacing = 2.7)
        ax.tick_params(labelsize=14)
        ax.set_box_aspect(None, zoom=0.80)
        plt.savefig("./raster_scan.png", bbox_inches='tight')
        self.raster_plot = QPixmap("./raster_scan.png")
        self.raster_label.setPixmap(self.raster_plot)
        self.save_button.setEnabled(True)
    def keyPressEvent(self, event: QKeyEvent):
        """
        This function is used for the keyboard control. Initiated whenever
        a key is pressed and then moves the fiber array if the one of the
        designated keys are pressed (Q, W, E, A, S, or D)
        """
        if self.keyboard_control_button.isChecked() == True:
            
            if event.key() == QtCore.Qt.Key.Key_W:
                if (self.setup.stepper.max_pos[0] - self.setup.position[3]) <= self.setup.stepper.jog_step_size[0]:
                    self.keyboard_control_message.setText("Movement out of bound.")
                    
                else:
                    self.setup.stepper.jog(1, 1) # Jog in positive x direction
                    self.keyboard_control_message.setText("   ")
            if event.key() == QtCore.Qt.Key.Key_S:
                if self.setup.position[3] <= self.setup.stepper.jog_step_size[0]:
                    self.keyboard_control_message.setText("Movement out of bound.")
                 
                else:
                    self.setup.stepper.jog(1, 2) # Jog in negative x direction
                    self.keyboard_control_message.setText("   ")
            if event.key() == QtCore.Qt.Key.Key_D:
                if (self.setup.stepper.max_pos[1] - self.setup.position[4]) <= self.setup.stepper.jog_step_size[1]:
                    self.keyboard_control_message.setText("Movement out of bound.")
                 
                else:
                    self.setup.stepper.jog(2, 1) # Jog in positive y direction
                    self.keyboard_control_message.setText("   ")
            if event.key() == QtCore.Qt.Key.Key_A:
                if self.setup.position[4] <= self.setup.stepper.jog_step_size[1]:
                    self.keyboard_control_message.setText("Movement out of bound.")
                 
                else:
                    self.setup.stepper.jog(2, 2) # Jog in negative y direction
                    self.keyboard_control_message.setText("   ")
            if event.key() == QtCore.Qt.Key.Key_E:
                if (self.setup.stepper.max_pos[2] - self.setup.position[5]) <= self.setup.stepper.jog_step_size[2]:
                    self.keyboard_control_message.setText("Movement out of bound.")
                 
                else:
                    self.setup.stepper.jog(3, 1) # Jog in positive yaw direction
                    self.keyboard_control_message.setText("   ")
                 
            if event.key() == QtCore.Qt.Key.Key_Q:
                if self.setup.position[5] <= self.setup.stepper.jog_step_size[2]:
                    self.keyboard_control_message.setText("Movement out of bound.")
                 
                else:
                    self.setup.stepper.jog(3, 2) # Jog in negative yaw direction
                    self.keyboard_control_message.setText("   ")
            pos = self.setup.stepper.get_positions() # Update positions in the window position boxes
            self.x_stepper_box.setValue(pos[0])
            self.y_stepper_box.setValue(pos[1])
            self.z_stepper_box.setValue(pos[2])
            self.setup.position[3:6] = pos
            
    def powselectionchange(self):
        """
        This function is initiated whenever the Power meter selection is 
        changed in the window.
        """
        if self.powcombobox.currentText() == "USB-6343 (BNC)":
            self.setup.DAQ = "NiDAQ"
            self.signal_label.setText("Coupling power [V]:")
        elif "MPM" in self.powcombobox.currentText():
            self.setup.DAQ = "MPM"
            self.signal_label.setText("Coupling power [dBm]:")
        elif self.powcombobox.currentText() == "Thorlabs powermeter":
            self.setup.DAQ = "TLPM"
            self.signal_label.setText("Coupling power [W]")
            
    def refresh_pow(self):
        """
        Intiated when the "Refresh" button is clicked. Checks if new 
        power meters are connected, and if so, adds them to the list
        """
        self.setup.refresh_pow()
        if self.setup.NiDAQ is not None and self.powcombobox.findText('NiDAQ') == -1:
            self.powcombobox.addItem('NiDAQ')
        if self.setup.mpm is not None and self.powcombobox.findText('Santec MPM-210') == -1:
            self.powcombobox.addItem('Santec MPM-210')
    def disable_channel(self, button, chan):
        if button.isChecked() == True:
            self.setup.stepper.disable_channel(chan)
            button.setText("Enable")
        else:
            self.setup.stepper.enable_channel(chan)
            button.setText("Disable")
    def home_stepper_channel(self, chan):
        self.setup.stepper.home(chan)
        box = [self.x_stepper_box, self.y_stepper_box, self.z_stepper_box][chan - 1]
        box.setValue(0)
    def zero_piezo_channel(self, axis):
        chan_map = {"x": 1, "y": 2, "z": 3}
        box = [self.x_box, self.y_box, self.z_box][chan_map[axis] - 1]
        box.setValue(0)
        self.setup.piezo.zero(axis)


    # def open_santec_gui(self):
    #     self.santec_window = Santec_GUI(rm = rm, listing=listing, system = system)
    #     self.santec_window.show()
    def home_all_controllers(self):
        self.setup.piezo.zero()
        self.setup.stepper.home_all()
        self.home_status_label.setText("Waiting for devices to zero and go into closed loop mode...")
        sleep(30)
        self.setup.set_position([15, 15, 30, 0, 0, 0])
        self.home_status_label.setText("Homing finished")
    def optimize(self, axes = [], step_sizes = [], method = "Pattern search"):
     
        if len(axes) == 0 and len(step_sizes) == 0:
            for i in range(6):
                if self.axis_check_boxes[i].isChecked():
                    step_sizes.append(float(self.step_size_boxes[i].text()))
                    axes.append(i)
            if len(axes) == 0:
                pass
            else:
                self.setup.optimize(axes = axes,
                                    method = self.optimize_method.currentText(),
                                    step_size = step_sizes, conv_tol = 0.8*0.8*np.min(step_sizes))
        else:
            self.setup.optimize(axes = axes,
                                method = method,
                                step_size = step_sizes, conv_tol = 0.8*0.8*np.min(step_sizes))
        self.update_position_boxes()
        
        
    def quick_optimize(self):
        """
        This function is a combination of optimzation steps that I have found useful
        It is first a pattern search using only the stepper motors, then followed by a
        hill climb using (xy) piezo twice, where second iteration is with smaller step size
        """
        step_sizes = [0.0005, 0.0005, 0.01]
        axes = [3, 4, 5]
        self.setup.optimize(axes = axes,
                            method = "Pattern search",
                            step_size = step_sizes, conv_tol = 0.8*0.8*np.min(step_sizes))
        step_sizes = [0.1, 0.1]
        axes = [0, 1]
        self.setup.optimize(axes = axes,
                            method = "Hill climb",
                            step_size = step_sizes, conv_tol = 0.8*0.8*np.min(step_sizes))
        step_sizes = [0.01, 0.01]
        axes = [0, 1]
        self.setup.optimize(axes = axes,
                            method = "Hill climb",
                            step_size = step_sizes, conv_tol = 0.8*0.8*np.min(step_sizes))
    def update_position_boxes(self, pos = None):
        if pos is None:
            for i, box in enumerate([self.x_box, self.y_box, self.z_box,
                        self.x_stepper_box, self.y_stepper_box, self.z_stepper_box]):
                box.setValue(self.setup.position[i])
        else:
            for i, box in enumerate([self.x_box, self.y_box, self.z_box,
                        self.x_stepper_box, self.y_stepper_box, self.z_stepper_box]):
                box.setValue(pos[i])
    def saveFunc(self):
        try:
            name = QFileDialog.getSaveFileName(self, 'Save File',"Pickle (*.pkl)")
            
            with open(name[0] + '.pkl', 'wb') as f:
                pickle.dump(self.setup.raster_data, f)
        except Exception as e:
            print("There is not data to save yet")
    def closeEvent(self, *args, **kwargs):
        self.signal_button.setChecked(False) # If this not done some error with the Santec powermeter connection occurs.
        if self.setup.simulate != True:
            self.setup.close()

app = QApplication(sys.argv)
window = MainWindow()    
if __name__ == '__main__':
    window.show() # IMPORTANT !!!
    app.exec()