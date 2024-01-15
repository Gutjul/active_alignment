# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:37:29 2023

@author: s176369
"""
import pyvisa
import numpy as np
import nidaqmx
import os
import time
import matplotlib.pyplot as plt
import random
from nidaqmx import stream_readers
from nidaqmx import stream_writers
from scipy.stats import multivariate_normal
from nidaqmx import constants
class Nidaq:
    def __init__(self, name= "Dev1", scan_rate = 1000, minV = [0, 0 ,0 ,0], 
                 maxV = [10, 10, 10, 10] , inputs=["ai0"], outputs=["ao0", "ao1", "ao2", "ao3"]):
        """
        A wrapper around the National Instruments USB-6343 Data Acquisition Device.
        Note that the device has more functionalities than the ones given here,
        and the script might be subject to extension.
        
        Parameters
        ----------
        name (str): Name/address of device. This can be found in the NI MAX software
        
        scan_rate (int): When performing scan using the daq_experiment function
                         the scan_rate speficies the number of voltage 
                         set/reads per second
        minV (1x4 float array): Specifies the minimum voltage of the four
                                analog output channels.
        maxV (1x4 float array): Specifies the minimum voltage of the four
                                analog output channels.
        inputs (str array): Specifies which analog input channels to use
        
        outputs (str array): Specifies which analog output channels to use 
        Returns
        -------
        None.
        
        """ 
        self.name = name
        self.outputs = outputs
        self.inputs = inputs
        self.maxV = np.array(maxV) # Set min and max to 10 and 0 respectively, because this is the voltage range accepted by the piezocontroller
        self.minV = np.array(minV)
        self.scan_rate = scan_rate
        # The DAQ_status.txt file containts the last setting analog outputs of the DAQ
        # This is useful if the software is terminated and the setting needs to be reloaded
        # in order for Python to know the voltage setting.
        # There might be a better way to do this
        self.status_location = os.path.realpath('nidaq.py').replace('nidaq.py', 'DAQ_status.txt')
        try:
            with open(self.status_location) as f:
                self.volt = np.array(eval(f.read()))
        except:
            print("DAQ status does not exist. Created new file")
            self.volt = np.array([0, 0, 0, 0])
            with open(self.status_location, 'w') as f:
                f.write(str(self.volt.tolist()))
    def set_volt(self, volt): #Set voltage of output channel typically "Dev1"
        """
        Sets the voltage of the four analog output channels
        Parameters
        ----------
        volt (1x4 float array): Desired voltage of the four analog output channels
        """
        volt = np.array(volt)
        if (abs(self.volt - volt) > 0.5).any(): # This check is made as a safety precursion to avoid big sudden changes in voltage
            raster_positions = self.create_settle_array(volt)
            self.daq_experiment(raster_positions, sr = 100)
        
        else:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.name+"/ao0")
                task.ao_channels.add_ao_voltage_chan(self.name+"/ao1")
                task.ao_channels.add_ao_voltage_chan(self.name+"/ao2")
                task.ao_channels.add_ao_voltage_chan(self.name + "/ao3")
                task.write(volt, auto_start=True)
        with open(self.status_location, 'w') as f: # Write status to DAQ_status.txt
            f.write(str(volt.tolist()))
        self.volt = np.array(volt)
        
    def read_input(self): 
        """
        Reads the voltage on the channel input channel
        
        """
        with nidaqmx.Task() as read_task:
            for i, input in enumerate(self.inputs):
                aichan = read_task.ai_channels.add_ai_voltage_chan(self.name + "/" + input)
                aichan.ai_min = 0
                aichan.ai_max = 10
            return read_task.read(number_of_samples_per_channel=1)[0]
        
        
    def read_input_long_continous(self, num_samples):
        """
        Reads a stream of voltage readings from the buffer of the analog input channel.
        
        
        
        Parameters
        ----------
        num_samples (int): Defines how many samples to read
        
        Returns
        ----------
        indata (1xnum_samples float array): The voltage readings
        
        """
        with nidaqmx.Task() as read_task:
            for i, input in enumerate(self.inputs):
                aichan = read_task.ai_channels.add_ai_voltage_chan(self.name + "/" + input)
                aichan.ai_min = 0
                aichan.ai_max = 10
            indata = np.zeros(num_samples)
            reader = stream_readers.AnalogSingleChannelReader(read_task.in_stream)
            reader.read_many_sample(indata, number_of_samples_per_channel= num_samples, timeout=10.0)

            return indata
        
    def read_input_long_discrete(self, num_samples, sr):
        """
        Same as read_input_long_continous, but here the readings are performed
        at a specific rate given by sr
        
        
        Parameters
        ----------
        num_samples (int): Defines how many samples to read
        
        sr (int): Rate at which the voltage samples are read
        Returns
        ----------
        indata (1xnum_samples float array): The voltage readings
        
        """
        with nidaqmx.Task() as read_task:
            for i, input in enumerate(self.inputs):
                aichan = read_task.ai_channels.add_ai_voltage_chan(self.name + "/" + input)
                aichan.ai_min = 0
                aichan.ai_max = 10
            read_task.timing.cfg_samp_clk_timing(rate=sr, source='OnboardClock', samps_per_chan=num_samples)
            indata = read_task.read(number_of_samples_per_channel=num_samples)
            return indata

    def daq_experiment(self, raster_positions, sr = None, delay = None, timeout = 100, num_samples =1):
        """
        This function is quite important.
        It takes a matrix (raster_positions) specifying a collection of settings
        for the analog outputs and then collects a corresponding number of voltage
        readings from the analog input
        
        Parameters
        ----------
        raster_positions (4xnum_samples float array): 
            Matrix defining the desired settings of the analog output
                          
        
        sr (int): Rate at which the voltages are set/read [1/s]
        
        delay (float): Specifies the time passing between a analog voltage is set
                       and the corresponding input voltage is read [s]
        
        timeout (float): Specifies how long to wait for the hardware to finish
                         before breaking the loop.
        Returns
        ----------
        indata (1xnum_samples float array): The voltage readings corresponding
                                            to raster_positions
        
        """
        if sr is None:
            sr = self.scan_rate
        if delay is None:
            delay = 1/sr
        
        if isinstance(raster_positions, np.ndarray) == False:
            raster_positions = np.array(raster_positions)

        if raster_positions.dtype != "float64":
            raster_positions = raster_positions.astype("float64")
        try:
            num_points = raster_positions.shape[1]+2
        except IndexError:
            self.set_volt(raster_positions)
            return self.read_input()
        
        with nidaqmx.Task() as read_task, nidaqmx.Task() as write_task:
            for i, output in enumerate(self.outputs):
                aochan = write_task.ao_channels.add_ao_voltage_chan(self.name + "/" + output)
                aochan.ao_min = 0
                aochan.ao_max = 10
            for i, input in enumerate(self.inputs):
                aichan = read_task.ai_channels.add_ai_voltage_chan(self.name + "/" + input)
                aichan.ai_min = 0
                aichan.ai_max = 10
            for task in (read_task, write_task):
                # task.timing.cfg_samp_clk_timing(rate=sr, source='OnboardClock', samps_per_chan=num_samples
                #                                 ,sample_mode= constants.AcquisitionType.CONTINUOUS)
                task.timing.cfg_samp_clk_timing(rate=sr, source='OnboardClock', samps_per_chan=num_points)
                #                                 ,sample_mode= constants.AcquisitionType.CONTINUOUS)
                
            write_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                read_task.triggers.start_trigger.term) # Set the trigger
            read_task.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS
            read_task.triggers.start_trigger.delay = delay # Set delay
            # Now save the volt to the file in case the scan fails
            self.volt = raster_positions[:, -1]
            with open(self.status_location,'w') as f:
                f.write(str(self.volt.tolist()))
            # Now a lot is going on down here. We found that some work arounds
            # have to be made in order to get the correct data, so some of
            # this code with deletion of variables and i similar might seem a bit confusing
            # but is nevertheless necessary
            raster_positions = np.concatenate((raster_positions[:, 0][np.newaxis].T, raster_positions,raster_positions[:, -1][np.newaxis].T),axis=1)
            write_task.write(raster_positions, auto_start=False)
            write_task.start()
            indata = np.asarray(read_task.read(number_of_samples_per_channel=num_points, timeout=timeout)).T
            indata = np.delete(indata, [0, -1], axis=0)  # Deletes value from last measurement which is still in the buffer
            return indata
    
    def spectrometer_sweep(self, chan = 0, start_volt = 0, stop_volt = 10, step_size = 0.1, scan_rate = 100, delay = 20.0e-9, timeout = 100):
        """
        This function is quite important.
        It takes a matrix (raster_positions) specifying a collection of settings
        for the analog outputs and then collects a corresponding number of voltage
        readings from the analog input
        
        Parameters
        ----------
        raster_positions (4xnum_samples float array): 
            Matrix defining the desired settings of the analog output
                          
        
        sr (int): Rate at which the voltages are set/read [1/s]
        
        delay (float): Specifies the time passing between a analog voltage is set
                       and the corresponding input voltage is read [s]
        
        timeout (float): Specifies how long to wait for the hardware to finish
                         before breaking the loop.
        Returns
        ----------
        indata (1xnum_samples float array): The voltage readings corresponding
                                            to raster_positions
        
        """

        sr = scan_rate / step_size # The scan rate in samples/second

        positions = self.create_sweep_array(chan, start_volt, stop_volt, step_size)
        with nidaqmx.Task() as read_task, nidaqmx.Task() as write_task:
            for i, output in enumerate(self.outputs):
                aochan = write_task.ao_channels.add_ao_voltage_chan(self.name + "/" + output)
                aochan.ao_min = 0
                aochan.ao_max = 10
            for i, input in enumerate(self.inputs):
                aichan = read_task.ai_channels.add_ai_voltage_chan(self.name + "/" + input)
                aichan.ai_min = 0
                aichan.ai_max = 10
            # write_task.timing.cfg_samp_clk_timing(rate = sr, source='OnboardClock', samps_per_chan = raster_positions.shape[1])
                #                                  ,sample_mode= constants.AcquisitionType.CONTINUOUS)

            # read_task.timing.cfg_samp_clk_timing(rate = sr * num_samples, source='OnboardClock', samps_per_chan=raster_positions.shape[1] * num_samples + 2)
                #                                  ,sample_mode= constants.AcquisitionType.CONTINUOUS)
            write_task.timing.cfg_samp_clk_timing(rate = sr, source='OnboardClock', samps_per_chan = positions.shape[1])
            read_task.timing.cfg_samp_clk_timing(rate = 500000, source='OnboardClock', samps_per_chan = int(positions.shape[1] * 500000 / sr) + 2)
            read_task.timing.delay_from_samp_clk_delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS
            read_task.timing.delay_from_samp_clk_delay = 20e-9
            
            # Set the trigger
            # read_task.triggers.start_trigger.delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS
            write_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                read_task.triggers.start_trigger.term)
            # read_task.triggers.start_trigger.retriggerable = True
            # read_task.triggers.start_trigger.delay = delay # Set delay
            # Now save the volt to the file in case the scan fails
            self.volt = positions[:, -1]
            with open(self.status_location,'w') as f:
                f.write(str(self.volt.tolist()))
            # Now a lot is going on down here. We found that some work arounds
            # have to be made in order to get the correct data, so some of
            # this code with deletion of variables and i similar might seem a bit confusing
            # but is nevertheless necessary
            # positions = np.concatenate((positions[:, 0][np.newaxis].T, positions, positions[:, -1][np.newaxis].T),axis=1)
            write_task.write(positions, auto_start=False)
            write_task.start()
            raw_data = np.asarray(read_task.read(number_of_samples_per_channel = int(positions.shape[1] * 500000 / sr) + 2, timeout=timeout)).T
            raw_data = np.delete(raw_data, [0, -1], axis=0)  # Deletes value from last measurement which is still in the buffer
            """Now create the averaged data where delay is taken into accound"""
            N = int(delay/(1 / 500000)) # Number output samples to remove per input to get right delay
            
            
            remove_indices = np.hstack(np.array([np.arange(int(i * 500000/sr), int(i * 500000/sr + N), 1) for i in range(positions.shape[1])]))
            avg_data = np.delete(raw_data, remove_indices)
                
            avg_data = np.mean(np.reshape(avg_data, (positions.shape[1], int(500000/sr - N))), axis = 1)


            
            return avg_data, raw_data
    
    
    def create_sweep_array(self, chan, start_volt, stop_volt, step_size):
        
        const_chans = np.setdiff1d([0, 1, 2, 3], chan) # Channels with constant voltage
        if start_volt < stop_volt:
            sweep_vals = np.arange(start_volt, stop_volt + step_size, step_size)
            positions = np.zeros([4, len(sweep_vals)])
        elif start_volt > stop_volt:
            sweep_vals = np.flip(np.arange(stop_volt, start_volt + step_size, step_size))
            positions = np.zeros([4, len(sweep_vals)])
        else:
            sweep_vals = start_volt
            positions = np.zeros([4, 1])
        positions[chan, :] = sweep_vals
        for const_chan in const_chans:
            positions[const_chan, :] = np.ones(len(sweep_vals)) * self.volt[const_chan]
        
        return positions

    
    def create_settle_array(self, volt):
        """
        This function creates and array of voltage settings in order to
        perform a smooth transition from current voltage to the set voltage.
        This is handy if we are working with very fine structures where
        big sudden voltage changes can collapse them.
        
        The created array can the be fed into the daq_experiment function
        to smoothly move to the desired voltage setting.
        
        """
        N = int(np.max(abs(self.volt - volt)) * 10 + 1) # Number of elements
        raster_positions = np.zeros([4, N])
        for i in range(4):
            if (abs(volt[i] - self.volt[i])) > 0.05:
                step = 0.1 * np.sign(volt[i] - self.volt[i])
                coords = np.arange(self.volt[i], volt[i], step)
                raster_positions[i, 0:len(coords)] = coords
                raster_positions[i, len(coords):N] = volt[i]
            else:
                raster_positions[i, :] = volt[i]
        return raster_positions

    


class Simulated_DAQ:
    def __init__(self, name="Dev1", scan_rate = 1000, minV = [0, 0 ,0 ,0], maxV = [10, 10, 10, 10] , inputs=["ai0"], outputs=["ao0", "ao1", "ao2", "ao3"]):
        self.name = name
        self.outputs = outputs
        self.inputs = inputs
        self.maxV = np.array(maxV) # Set min and max to 10 and 0 respectively, because this is the voltage range accepted by the piezocontroller
        self.minV = np.array(minV)
        self.scan_rate = scan_rate
        # Create a simulated distribution to optimize

        self.cov = [[1, 0, 0], [0, 1, 0], [0, 0, 5]]
        self.mean = [5, 5, 0]
        self.piezo_calibration = np.array([4.665, 4.6157, 4.064])
        
        self.dist = multivariate_normal(self.mean, self.cov)
        self.volt = [0, 0, 0, 0]
        # self.dist = []
    
    def read_input(self, pos = [15, 15, 15, 0]):
        return self.dist.pdf([pos[0]/self.piezo_calibration[0], pos[1]/self.piezo_calibration[1], pos[2]/self.piezo_calibration[2]])
        # return self.dist.pdf(self.position)
        
    def set_volt(self, pos):
        pass
    
    def daq_experiment(self, positions, sr = 300, timeout = 5000):
        return self.dist(positions)
        
    def read_input_long_continous(self, num_samples):
        return self.dist.pdf(self.position)
    
    
 