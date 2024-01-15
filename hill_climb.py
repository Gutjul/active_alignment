# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:06:18 2023

@author: s176369
"""
import numpy as np
from time import sleep

class Hill_Climb():
    def __init__(self, setup, axes = [0, 1, 2, 3, 4, 5],
                 step_size = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                 num_samples = 50, settle_time = 0.1):
        self.axes = np.array(axes)
        self.bounds = np.array([[0, 30], [0, 30], [0, 30], [0, 6], [0, 6], [0, 6]])[self.axes]
        self.step_size = np.array( [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.step_size[self.axes] = step_size
        

        self.setup = setup
        self.position = np.array(self.setup.position, dtype='float64')
        self.num_samples = num_samples
        self.settle_time = settle_time
        sleep(self.settle_time)
        self.max_volt = np.mean(self.setup.read_input_long(self.num_samples))
        self.volt = self.max_volt
    
    def climb_axis(self, axis, step_size, direction):
        while self.check_bound(axis, self.position[axis] + direction * self.step_size[axis]):
            
            self.position[axis] = self.position[axis] + direction * self.step_size[axis]
            self.setup.set_position(self.position)
            self.volt = np.mean(self.setup.read_input_long(self.num_samples))
            if self.volt > self.max_volt:
                self.max_volt = self.volt
            else:
                # climb back again
                self.position[axis] = self.position[axis] - direction * self.step_size[axis]
                self.setup.set_position(self.position)
                sleep(self.settle_time)
                self.volt = np.mean(self.setup.read_input_long(self.num_samples))
                break
    def iterate(self):
        self.max_volt = np.mean(self.setup.read_input_long(self.num_samples))
        for axis in self.axes:
            direction = self.check_direction(axis, self.step_size[axis])
            if direction != 0:
                self.climb_axis(axis, self.step_size[axis], direction = direction)
            
        return self.volt
    def iterate_cont(self):
        
        self.max_volt = np.mean(self.setup.read_input_long(self.num_samples))
        while self.step_size[5] >= 0.005:
            old_volt = self.volt
            new_volt = self.iterate()
            if old_volt > new_volt:
                self.step_size[5] = self.step_size[5] * 0.8
        return self.volt
        
    def check_direction(self, axis, step_size):
        self.position[axis] = self.position[axis] + 1 * self.step_size[axis]
        self.setup.set_position(self.position)
        sleep(self.settle_time)
        self.volt = np.mean(self.setup.read_input_long(self.num_samples)) # This number could possible benefit from high number of samples
        if self.volt > self.max_volt:
            self.max_volt = self.volt
            return 1
        self.position[axis] = self.position[axis] - 2 * 1 * self.step_size[axis]
        self.setup.set_position(self.position)
        sleep(self.settle_time)
        self.volt = np.mean(self.setup.read_input_long(self.num_samples))
        if self.volt > self.max_volt:
            self.max_volt = self.volt
            return -1
        self.position[axis] = self.position[axis] + 1 * self.step_size[axis]
        self.setup.set_position(self.position)
        sleep(self.settle_time)
        self.volt = np.mean(self.setup.read_input_long(self.num_samples))
        return 0
        
            
    def check_bound(self, axis, position):
        if position < self.bounds[axis, 0]:
            return False
        if position > self.bounds[axis, 1]:
            return False
        else:
            return True
        
        
    
        
        
        
        
        
        

