# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:44:13 2023

@author: s176369
"""
from msl.equipment.resources.thorlabs.kinesis.benchtop_stepper_motor import BenchtopStepperMotor
from msl.equipment.resources.thorlabs.kinesis.api_functions import Benchtop_StepperMotor_FCNS
from msl.equipment.resources.thorlabs import MotionControl
MotionControl.build_device_list()
from msl.equipment import (
    EquipmentRecord,
    ConnectionRecord,
    Backend,
    exceptions
)
import numpy as np
import raster_scan
from matplotlib import cm
import matplotlib.pyplot as plt
class BSC203(BenchtopStepperMotor):
    def __init__(self, serial_no = '70391704', axes = [1, 2, 6], jog_step_size = [0.01, 0.01, 0.15]):
        """A wrapper around ``Thorlabs.MotionControl.Benchtop.StepperMotor.dll``.

        The :attr:`~msl.equipment.record_types.ConnectionRecord.properties`
        for a BenchtopStepperMotor connection supports the following key-value pairs in the
        :ref:`connections-database`::

            'device_name': str, the device name found in ThorlabsDefaultSettings.xml [default: None]

        Do not instantiate this class directly. Use the :meth:`~.EquipmentRecord.connect`
        method to connect to the equipment.

        
        Jesper Sand:
            This is an upgrade I made of the BenchtopStepperMotor class made by some guys at Thorlabs.
            I modified it to be more specific for the BSC203 controller and added some functions
            that I thought would be practical. 
            To see the remaining functions that are part of this class go to: 
            /--your env--/Lib/site-packages/msl/equipment/resources/thorlabs/kinesis
            and open benchtop_stepper_motor.py to inspect.
            
        Parameters
        ----------
        serial_no (str): Serial number of the device.
        Can be found by opening the Kinesis software
        
        axes are given to load the correct settings for the given axes. 
        The encoding is as follows:
            1: X
            2: Y
            3: Z
            4: Roll (rotation around X axis)
            5: Pitch (rotation around Y axis)
            6: Yaw (rotation around Z axis)
        
        jog_step_size (double array): Specifies jog step size of channels 1-3.
        """
        # The following line specifies which device options to load for the 
        # NanoMax is the stage and DRV208 is the name of the stepper motors
        self.axes = axes
        self.named_settings = ["NanoMax 600 X Axis (DRV208)", "NanoMax 600 Y Axis (DRV208)", 
                               "NanoMax 600 Z Axis (DRV208)", "NanoMax 600 Roll Axis (DRV208)",
                               "NanoMax 600 Pitch Axis (DRV208)", "NanoMax 600 Yaw Axis (DRV208)"]
        # Defines record, which has to be done for this instrument.
        record = EquipmentRecord(
            manufacturer='Thorlabs',
            model='BSC203',  # update for your device BSC101
            serial=serial_no,  # update for your device
            connection=ConnectionRecord(
                address='SDK::Thorlabs.MotionControl.Benchtop.StepperMotor.dll',
                backend=Backend.MSL,
            )
        )
        # Connect
        super(BenchtopStepperMotor, self).__init__(record, Benchtop_StepperMotor_FCNS)
        
        self._num_channels = self.get_num_channels()
        
        
        # The following 6 lines loads the settings corresponding to the unit
        for i in range(3):
            self.load_named_settings(i + 1, self.named_settings[axes[i] - 1])
            self.load_settings(i + 1)
        
        
        

        
        self.get_calibration_numbers() # Retrieve calibration constants to map from device units to SI units
        self.integer_positions = [int(self.get_position(i + 1) / self.calibration_number[self.axes[i] - 1]) for i in [0, 1, 2]] # Position in device units
        
        self.jog_step_size = jog_step_size # Jog step size when initiating class
        for i in range(3):
            self.set_jog_step_size_SI(i + 1, self.jog_step_size[i]) # Set jog step size in instrument
            self.set_jog_vel_params(i + 1, 43980465, 50000) # Set velocity parameter. 2nd argument is velocity and 3rd is acceleration in device units
            self.set_jog_mode(i + 1, "SingleStep", "Immediate") # Set jog mode

    def set_position(self, pos, wait = True):
        """
        Parameters
        ----------
        pos : double array of length 3
            Position to move to given in mm.

        Returns
        -------
        None.
        
        Same functionality as move_to_position, but where the code waits for 
        the motors to finish their movement. This will be handy in some situations.
        """
        for i, chan in enumerate([1, 2, 3]):
            self.integer_positions[i] = int( pos[i] * self.calibration_number[self.axes[i] - 1] )
            self.move_to_position(chan, self.integer_positions[i])
        if wait == True:
            for i, chan in enumerate([1, 2, 3]):
                while self.get_position(chan) != self.integer_positions[i]:
                    continue
    def set_relative_position(self, dist):
        """
        Parameters
        ----------
        pos : double array of length 3
            Position to move to given in mm.

        Returns
        -------
        None.
        
        Same functionality as move_to_position, but where the code waits for 
        the motors to finish their movement. This will be handy in some situations.
        """
        for i, chan in enumerate([1, 2, 3]):
            dist_integer = int(dist[i] * self.calibration_number[self.axes[i] - 1])
            self.integer_positions[i] = self.integer_positions[i] + dist_integer
            self.move_relative(chan, dist_integer)
        # if wait == True:
        #     for i, chan in enumerate([1, 2, 3]):
        #         while self.get_position(chan) != self.integer_positions[i]:
        #             continue

        
    def get_positions(self):
        """
        
        Description
        -----------
        Retrieves the positions from the stepper motor.

        ----------
        None.
        
        Returns
        -------
        An array corresponding to the positions of the 3 channels.
        The returned position is given in SI units

        """
        positions = np.zeros(3)
        for i in [0, 1, 2]:
            positions[i] = self.get_position(i + 1)/self.calibration_number[self.axes[i] - 1]
        return positions
        
    def jog(self, chan, direction, wait = True):
        """
        
        Description
        -----------
        Moves the motor a distance specified by self.stepper.jog_step_size.
        
        Parameters
        ----------
        chan (integer): The channel to jog (1 to 3)
        
        direction (integer): 1 is forward and 2 is backwards
        
        wait (boolean): If wait is True, the code while wait for the motor
                        to finish the move. This is generally recommended
        
        Returns
        -------
        None.
        
        
        """
        if direction == 1:
            new_pos = self.get_position(chan) + self.jog_step_size_int[chan - 1]
        elif direction == 2:
            new_pos = self.get_position(chan) - self.jog_step_size_int[chan - 1]
        self.move_jog(chan, direction)
        if wait == True:
            while self.get_position(chan) != new_pos:
                continue

    def get_calibration_numbers(self):
        """"
        
        This function gets calibration numbers that can be used to translate
        between integer position and the physical position given in milimeters
        and degrees.
        It is simply found by retrieving the mininum and maximum allowed
        integer positions from the device, and then comparing it to the
        min/max given in mm/degrees, which gives the proper calibration constants.
        
        """
        self.calibration_number = np.zeros(6)
        self.min_pos = np.zeros(3)
        self.min_pos_int = np.zeros(3)
        self.max_pos = np.zeros(3)
        self.max_pos_int = np.zeros(3)
        self.jog_step_size_int = [self.get_jog_step_size(1), self.get_jog_step_size(2), self.get_jog_step_size(3)]
        self.jog_step_size = np.zeros(3)
        for i in range(3):
            pos = self.get_motor_travel_limits(i + 1)
            self.min_pos[i] = pos[0]
            if i == 3:
                self.min_pos_int[i] = 0
            else:
                self.min_pos_int[i] = self.get_stage_axis_min_pos(i + 1)
            self.max_pos[i] = pos[1]
            self.max_pos_int[i] = self.get_stage_axis_max_pos(i + 1)
            self.calibration_number[self.axes[i] - 1] = (self.max_pos_int[i] - self.min_pos_int[i]) / (self.max_pos[i] - self.min_pos[i])
    
    def set_jog_step_size_SI(self, chan, step_size):
        """    
        Parameters
        ----------
        chan (integer): The channel
        jog_step_size (double): the jog_step_size given in SI units (mm or degrees)
            
        Returns
        -------
        None.
        """
        self.jog_step_size_int[chan - 1] = int(step_size * self.calibration_number[self.axes[chan - 1] - 1])
        self.set_jog_step_size(chan, int(self.jog_step_size_int[chan - 1]))
    def set_jog_step_size(self, chan, step_size):
        
        """Sets the distance to move on jogging.

        See :meth:`get_device_unit_from_real_value` for converting from a
        ``RealValue`` to a ``DeviceUnit``.

        Parameters
        ----------
        channel : :class:`int`
            The channel number (1 to n).
        step_size : :class:`int`
            The step size in ``DeviceUnits`` (see manual).

        Raises
        ------
        ~msl.equipment.exceptions.ThorlabsError
            If not successful.
        """
        self.jog_step_size[chan - 1] = step_size/self.calibration_number[self.axes[chan - 1] - 1]
        self.sdk.SBC_SetJogStepSize(self._serial, self._ch(chan), step_size)
    def home_all(self):
        """
        Homes all channels, that is, moving to a known position in order
        to correctly locate the position of the steppers.
        This should be done whenever the instrument is
        powered up. Furthmore, a homing is required when a channel has been
        disabled and enabled again, in order for the instrument to know the
        physical location
        """
        for chan in [1, 2, 3]:
            self.home(chan)

            
