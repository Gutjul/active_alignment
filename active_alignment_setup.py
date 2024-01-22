# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:33:23 2023

@author: s176369
"""
import numpy as np
from Thorlabs_BPC303.Thorlabs_BPC303 import bpc303
from bsc203 import BSC203
from time import sleep
from nidaq import Nidaq
from nidaq import Simulated_DAQ
import scipy.optimize
import numpy as np
import time
from custom_simplex import simplex_algorithm_new
import raster_scan
from hill_climb import Hill_Climb
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
import serial
from thorlabs_powermeter import Powermeter
from pattern_search import pattern_search
import pyvisa
import nidaqmx
import Santec_IL_STS
from santec_communication import Santec_MPM
rm=pyvisa.ResourceManager()
listing=rm.list_resources() #creat a list of all detected connections
system=nidaqmx.system.System.local()

class Active_Alignment_Setup():
    def __init__(self, piezo_ID = '71345664', stepper_ID = '70391704', DAQ = "NiDAQ", 
                 simulate = False, zero = False, stepper_axes = [1, 2, 6]):
        """
        This function puts together the steppermotor controller, piezo controller,
        and a handful of power meters.
        The functions in this script is only for performing actions that
        require more than one instrument (for example setting a position with
        one of the controllers and reading a power with a power meter)
        
        Parameters
        ----------
        piezo_ID : :class:`str`
            The address of the piezo controller. Can be found in the Kinesis Software
        stepper_ID : :class:`str`
            The address of the stepper motor controller. Can be found in the Kinesis Software
            
        DAQ : :class:`str`
            Instructs what type of spectrometer to use (NiDAQ: National Instruments DAQ,
                                                        MPM: Santec power meter, 
                                                        TLPM: Thorlabs power meter)
        simulate : :class:'boolean'
            If True the instruments will not be initiated but a simple object
            that takes position inputs and return an output if initialized
            This option was handy at the development stage, but is not of 
            much use anymore
            
        zero: : :class:`boolean`
            Instructs whether to zero the instruments on initialization. 
            This is only necessary when the instruments have been turned off
            
        stepper_axes : :class: int array
            Instructs which axes are connected to the stepper motor.
            Mapping is the following:
                x: 1
                y: 2
                z: 3
                roll (rotation around x): 4
                pitch (rotation around y): 5
                yaw (rotation around z): 6
            Default is [1, 2, 6] meaning that 'x' (1) is connected to channel 1,
            'y' (2) is connected to channel 2, and 'yaw' (6) is connected to channel 3
        """
        self.DAQ = DAQ
        self.simulate = simulate
        self.position = np.zeros(6)
        self.stepper_axes = stepper_axes
        
        self.piezo_calibration = np.array([4.645, 4.616, 4.707]) # Numbers to convert from voltage on DAQ to physical position given in microns/V
        if self.simulate == False:
            self.piezo = bpc303.BPC303(piezo_ID)
            self.stepper = BSC203(stepper_ID, axes = self.stepper_axes)
            try:
                self.piezo.connect()
            except:
                print("Piezo not connected")
            if zero == True:
                self.piezo.zero()
                self.stepper.home_all()
                
                
                print("Waiting for devices to zero and go into closed loop mode...")
                sleep(30)
                self.set_position([0, 0, 0, 0, 0, 0])

            self.set_piezo_position([15 ,15, 30])
            self.position = [15, 15, 30, self.stepper.get_positions()[0], 
                             self.stepper.get_positions()[1], self.stepper.get_positions()[2]]
            self.find_power_meters()
            for i, name in enumerate(self.power_readers):
                if "MPM" in name:
                    try:
                        self.mpm = Santec_MPM(GPIBboard = 0, address = self.power_reader_address[i])
                        self.mpm.connect_mpm()

                    except:
                        print("Santec power meter not connected")
                if 'USB-6343 (BNC)' in name:
                    try:
                        self.NiDAQ = Nidaq(self.power_reader_address[i])
                    except:

                        print("National Instruments DAQ not connected")
                    
                if "Thorlabs powermeter" in name:
                    try:
                        self.tlpm.Connect(self.power_reader_address[i])
                    except:
                        print("Thorlabs powermeter not connected")
                        
            try:
                self.mpm
            except:
                self.mpm = None
            try:
                self.NiDAQ
            except:
                self.NiDAQ = None
            try:
                self.tlPM
            except:
                self.tlPM = None    
            # self.stepper.set_direction(1, False) # COMMENTED THIS LINE. IF SOMETHING GOES WRONG TRY AND INCLUDE AND SEE IF IT FIXES THINGS
        else:
            self.piezo = []
            self.DAQ = "Simulated DAQ"
            self.sim_daq = Simulated_DAQ()
            self.stepper = []
            # Define multivariate distribution in case we are simulating a signal
            self.cov = [[5, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 2], [0, 0, 5, 0, 0, 0],
                        [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0], [0, 2, 0, 0, 0, 2]]
            self.mean = [15, 15, 15, 2, 2, 3]
            self.dist = multivariate_normal(self.mean, self.cov)
            

    
    def optimize(self, axes = [3, 4, 5], method = 'Pattern search', conv_tol = 0.01*0.8*0.9, step_size = [0.0005, 0.0005, 0.01], num_samples = 100):
        """
        This function executes an optimization algorithm that optimizes the
        coupling through the circuit.

        Parameters
        ----------
        axes (float array),
            Specifies the axes to optimize according the following mapping:
                0: x, 1: y, 2: z, 4: roll, 5: pitch, 6: yaw
        method (str),
            Specifies the optimization algorithm.
            Possible values:
                Pattern search: The Hooke-Jeeves pattern search algorithm
                that has been claimed to work well for optical alignment
                
                Hill climb
                
                Dichotomy: An iterative version of Hill climb where the step size
                is reduced for each iteration
        conv_tol (float): Only relevant for pattern search. Specifies the minimum step size
                          before optimization terminates.
                          
        step_size (1xlen(axes) float array): 
            Specifies the initial step size of the chosen axes
            
        num_samples (int): 
            Specifies the number of samples acquired from the power meter
            for each measurement. Increasing this parameter will reduce noise
            but make the whole scheme slower.
        Returns
        -------
        Final coupling power after optimization

        """
        step_size = np.array(step_size)
        if method == 'Pattern search':
            # Do pattern search with stepper motors
            algorithm = pattern_search(self, axes = axes, conv_tol = conv_tol, step_size = step_size, num_samples = num_samples)
            return algorithm.iterate()
        if method == "Dichotomy":
            algorithm = Hill_Climb(self, axes = axes,
                                  step_size = step_size, settle_time = 0.0)
            current_res = self.read_input()
            new_res = current_res + 1
            while  new_res >= current_res:
                current_res = new_res
                new_res = algorithm.iterate()
            algorithm = Hill_Climb(self, axes = axes,
                                  step_size = step_size*0.1, settle_time = 0.0)
            new_res = current_res + 1
            while  new_res >= current_res:
                current_res = new_res
                new_res = algorithm.iterate()
            return new_res
        if method == "Hill climb":
            algorithm = Hill_Climb(self, axes = axes,
                                  step_size = step_size, settle_time = 0.0)
            return algorithm.iterate()
        
    def set_position(self, pos):
        """
        Sets position of piezo and stepper motor

        Parameters
        ----------
        pos (1x6 float array): First 3 values are for the 3 piezo channels
                               and the remaining 3 are for the stepper channels

        """
        if self.simulate == False:
            self.piezo.set_position(pos[0], pos[1], pos[2])
            self.set_stepper_position([pos[3], pos[4], pos[5]])
            self.position = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
        else:
            self.position = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
    def set_piezo_position(self, pos):
        """
        Sets position of piezo

        Parameters
        ----------
        pos (1x3 float array): Positions of the 3 piezo channels (usually x, y, and z)

        """
        if self.simulate == False:
            self.piezo.set_position(pos[0], pos[1], pos[2])
        self.position[0:3] = [pos[0], pos[1], pos[2]]

    def set_stepper_position(self, pos, wait = True):
        if self.simulate == False:
            try:
                self.stepper.set_position(pos, wait = wait)
                self.position[3:6] = [pos[0], pos[1], pos[2]]
            except: # This exception is made if communication fails with stepper controller
                    # I believe I have elimated bugs that could lead to this
                    # but have chosen to keep this part just in case 
                self.stepper = BSC203(axes = self.stepper_axes)
                for i in range(3):
                    self.stepper.set_jog_size(i + 1, self.stepper.jog_step_size_int[i])
        self.position[3:6] = [pos[0], pos[1], pos[2]]
        
    def read_input(self):
        """
        Reads the input from the current chosen power meter.
        Doing it this way makes it easy to implement another powermeter
        
        Returns
        ----------
        Coupling power in V, dBm, or W depending on choise of power meter

        """
        if self.simulate == False:
            if self.DAQ == "NiDAQ":
                return self.NiDAQ.read_input()
            elif self.DAQ == "MPM":
                return self.mpm.read_power()
            elif self.DAQ == "TLPM":
                self.tlpm.MeasurePower()
                return self.tlpm.power.value
            else:
                print("No power reader connected")
        else:
            return self.sim_daq.read_input(self.position)

    def read_input_long(self, num_samples):
        """
        Reads the input from the current chosen power meter by avearing a 
        number of samples. This way noise can be reduced
        
        
        Paramters
        ----------
        num_samples (int): Number power readings to average
        
        Returns
        ----------
        Coupling power in V, dBm, or W depending on choise of power meter

        """
        if self.simulate == False:
            if self.DAQ == "NiDAQ":
                return np.mean(self.NiDAQ.read_input_long_continous(num_samples))
            elif self.DAQ == "MPM":
                return self.mpm.read_power()
            elif self.DAQ == "TLPM":
                self.tlpm.SetAvgTime(num_samples/1000)
                self.tlpm.MeasurePower()
                self.tlpm.SetAvgTime(0.001)
                return self.tlpm.power.value
        else:
            return self.dist.pdf(self.position)
    def run_piezo_raster(self, width = [30, 30], step_size = 0.1, axes = [0, 1], plot = False):
            
    # def run_piezo_raster(self, width = [30, 30], step_size = 0.1, axes = [0, 1], plot = False):
    #     center = self.position[axes]
    #     const_axis = np.setdiff1d([0, 1, 2], axes)
    #     const_pos = self.position[const_axis]
    #     # Now convert from position to a voltage on the DAQ
        
    #     raster_scan.single_raster_scan(self.dq, center, width, step_size, axes, [const_volt, 0], simulate = False)
        
        
        """
        Performs a raster scan using the NiDAQ outputs connected to the piezo 
        controller channels.
        I have not included this function in the GUI, since the NiDAQ will not
        be placed in the setup per default.
        However, this function is handy for measuring alignment tolerances of grating 
        couplers and might be subject for further use. Hence the function
        is kept here.
        """
        if self.DAQ == "NiDAQ":
            init_position = np.array(self.position[0:3]) # Initial position before raster scan
            center = np.divide(init_position[axes], self.piezo_calibration[axes])
            width = np.array(width)
            width = np.divide(np.array(width), self.piezo_calibration[axes]) #
            
            const_axis = np.setdiff1d([0, 1, 2], axes)[0]
            const_pos = self.position[const_axis]
            # print("Center: " + str(center))
            # print("Width: " + str(width))
            # print("const axis: " + str(const_axis))
            # print("const_pos: " + str(const_pos))
            init_position[axes] = [0, 0]
            self.set_piezo_position(init_position) # Set the scanning axes at 0
             
            raster_results, raster_positions, local_max, max_val, Z, X, Y = raster_scan.single_raster_scan(
                self.NiDAQ, center, width, step_size, axes, [0, 0], self.simulate)
            self.NiDAQ.set_volt([0, 0, 0, 0])
            # Now translating from voltages on DAQ to physical positions
            X = X * self.piezo_calibration[axes[0]]
            Y = Y * self.piezo_calibration[axes[1]]
            local_max[0] = self.piezo_calibration[axes[0]]*local_max[0]
            local_max[1] = self.piezo_calibration[axes[1]]*local_max[1]
            end_position = init_position
            end_position[axes] = local_max
            end_position[const_axis] = const_pos
            self.set_piezo_position(end_position)
            self.raster_data = {'X': X, 'Y': Y, 'Z': Z}
            return raster_results, raster_positions, local_max, max_val, Z, X, Y
        else:
            print("Current DAQ has to be USB-6343 (NiDAQ) to perform piezo raster scan")

    def run_stepper_raster(self, width = [0.1, 0.1], step_size = 0.01, plot = False):
        """
        Function to run a raster scan with the stepper motor. This can be 
        used to locate the position at which coupling occurs.
        The movements are based on the "jog" feature of the controller, which
        proved to be faster than simply setting the specific position for each
        movement.
        
        Parameters:
        ----------
        width (1x2 float array):
            Specifies the width of the scan in the x and y directions in mm.
            Default is 0.1 which is sufficient to locate the peak if
            alignment marks are used.
            
        step_size (float):
            Specifies the step size of the scan (or resolution) in mm. Default is 0.01
            and it is not advised to go higher than that.
            
        plot (boolean):
            Specifies whether to plot results or not 
        Return:
        ---------
        X, Y, Z - matrices used for 3D plotting
        """
        for i in range(2):
            self.stepper.jog_step_size[i] = step_size
            self.stepper.set_jog_step_size_SI(i + 1, self.stepper.jog_step_size[i])
        init = self.stepper.get_positions() # Get the position of the stepper
        center = init[0:2] # Center of the x, y scan
        yaw = init[2] # The yaw coordinate that is fixed
        xi = center[0] - width[0]/2 if center[0] - width[0]/2 >= 0 else 0
        xf = center[0] + width[0]/2 if center[0] + width[0]/2 <= self.stepper.max_pos[0] else self.stepper.max_pos[0]
        yi = center[1] - width[1]/2 if center[1] - width[1]/2 >= 0 else 0
        yf = center[1] + width[1]/2 if center[1] + width[1]/2 <= self.stepper.max_pos[1] else self.stepper.max_pos[1]
        
        if xi == xf:
            positions_x = np.array([xi])
        else:
            positions_x = np.arange(xi, xf, self.stepper.jog_step_size[0])
        if yi == yf:
            positions_y = np.array([yi])
        else:
            positions_y = np.arange(yi, yf, self.stepper.jog_step_size[1])
            
        self.set_stepper_position([xi,  yi, yaw]) # Put FA at initial position
        
        n_x = len(positions_x) # Number of jogs in x direction
        n_y = len(positions_y) # Number of jogs in y direction
        readings = np.zeros([n_x, n_y]) # Initiate matrix for power readings
        for i in range(n_x):
            if i % 2 == 0: # Every second iteration executes this in order to perform snake-like movement
                if i != 0: # For first jog we dont move in x direction but scan along xi
                    self.stepper.jog(1, 1) # Jog in x-direction
                    self.position[3] += self.stepper.jog_step_size[0] # save position
                readings[i, 0] = self.read_input() # read power
                for j in range(1, n_y):
                    self.stepper.jog(2, 1) # Jog in y-direction
                    self.position[4] += self.stepper.jog_step_size[1] # save position
                    readings[i, j] = self.read_input() # read power
            else:
                self.stepper.jog(1, 1) # jog in x- direction
                self.position[3] += self.stepper.jog_step_size[0] # save position 
                readings[i, 0] = self.read_input() # read power
                for j in range(1, n_y):
                    self.stepper.jog(2, 2) # Jog in y direction
                    self.position[4] -= self.stepper.jog_step_size[1] # save position
                    readings[i, j] = self.read_input() # read power
        X, Y = np.meshgrid(positions_x, positions_y) # The usual 3D plotting procedure
        Z = raster_scan.flip_every_second_row(readings) # Flip every second row, because the scan moved in snake-like manner
        ind = np.unravel_index(np.argmax(Z, axis=None), Z.shape) # Find indices of maximum in x and y
        if np.max(Z)/np.min(Z) < 1.3: # Check if the measured scan is only noise
            self.set_stepper_position([center[0], center[1], yaw]) # if so, go back to initial position before scan
        else: # Else put FA at maximum position
            self.set_stepper_position([positions_x[ind[0]], positions_y[ind[1]], yaw])
        self.raster_data = {'X': X, 'Y': Y, 'Z': Z}
        if plot == True:
            
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, Z.T, cmap=cm.coolwarm,
                                    linewidth=0, antialiased=False)
            ax.set_xlabel("X position [microns]", fontsize=10, rotation=0)
            ax.set_ylabel("Y position [microns]", fontsize=10, rotation=0)
            ax.set_zlabel("Photodetector voltage [V]", fontsize=10, rotation=0)
        return X, Y, Z

    def find_power_meters(self):
        """
        Locates connected power meters and retrieves their addresses wihtout connecting
        """
        self.power_readers = []
        self.power_reader_address = []
        for name in system.devices.device_names: # Find NiDAQ name
            if nidaqmx.system.device.Device(name).product_type == "USB-6343 (BNC)":
                self.power_readers.append(nidaqmx.system.device.Device(name).product_type)
                self.power_reader_address.append(name)
        # Now for the Santec powermeter
        tools=[i for i in listing if 'GPIB' in i]
        tools = listing
        for i in range(len(tools)):
            #connect GPIB into a buffer
            try:
                buffer=rm.open_resource(tools[i], read_termination='\r\n')
                string = buffer.query('*IDN?')
                if "MPM" in string:
                    self.MPM = string
                    self.power_readers.append(string)
                    self.power_reader_address.append(listing[i])
            except:
                pass
        # Now add the Thorlabs powermeter
        try:
            self.tlpm = Powermeter()
            self.power_reader_address.append(self.tlpm.GetAddress())
            self.power_readers.append("Thorlabs powermeter")
        except:
            pass
        
        
    def refresh_pow(self):
        try: 
            self.mpm.read_input() # Try to get some response, if None or disconnected this one gives an error
        except:
            try:
                self.mpm = Santec_MPM(GPIBboard = 0, address = 16)
                self.mpm.connect_mpm()
            except:
                self.mpm = None
                print("Santec power meter not connected")
        
        if self.NiDAQ is None:
            try:
                address = self.setup.power_reader_address[
                    np.where(np.array(self.setup.power_readers) == 'USB-6343 (BNC)')[0][0]]
                self.NiDAQ = Nidaq(address)
            except:
                print("NiDAQ not connected")
        if self.tlpm is None:
            try:
                self.tlpm = Powermeter()
                self.tlpm.Connect(self.tlpm.GetAddress())
            except:
                print("Thorlabs power meter not connected")
        
    def close(self):
        if self.piezo is not None:
            self.piezo.shutdown()
        if self.stepper is not None:
            self.stepper.close()
        if self.tlpm is not None:
            self.tlpm.ClosePwrMtr()
        if self.mpm is not None:
            self.mpm.disconnect()