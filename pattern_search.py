# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:32:51 2023

@author: s176369
"""
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm
from nidaq import Nidaq
#matplotlib.use('Qt5Agg')


class pattern_search:
    
    def __init__(self, setup, axes = [3, 4, 5], step_size = [0.0005, 0.0005, 0.025], conv_tol = 0.02, num_samples = 50):
        if len(axes) != len(step_size):
            print("Optimization axes and step_size array must have same length")
        self.setup = setup
        self.axes = np.array(axes)
        self.position = np.array(self.setup.position)
        self.bounds = np.array([[0, 30], [0, 30], [0, 30], [0, 4], [0, 4], [0, 6]])
        self.step_size = np.array(step_size)

        self.conv_tol = conv_tol
        if self.setup.mpm is not None:
            self.setup.mpm.set_averaging_time(10)
        self.num_samples = num_samples
    
    
    def iterate(self):
        
        R = np.array(self.position, dtype = float)
        R = self.check_bounds(R)
        self.setup.set_position(R)
        Rval = self.setup.read_input_long(self.num_samples)
        X = R.copy()
        Xval = Rval
        while np.min(self.step_size) >= self.conv_tol:
            print(np.min(self.step_size))
            # Find next pattern
            Xold = X.copy()
            X, Xval = self.direction_search(X, Xval)
            X = self.check_bounds(X)
            self.setup.set_position(X)
            Xval = self.setup.read_input_long(self.num_samples)
            if (Xold == X).all():
                #self.step_size[2] = self.step_size[2] * 0.8
                self.step_size *= 0.8
                #self.step_size = self.step_size * 0.8
            else:
                if Rval < Xval:
                    Rold = R.copy()
                    R = X.copy()
                    Rval = Xval
                    X[self.axes] = 2 * R[self.axes] - Rold[self.axes]
                    X = self.check_bounds(X)
                    self.setup.set_position(X)
                    Xval = self.setup.read_input_long(self.num_samples)
                else:
                    #self.step_size[2] = self.step_size[2] * 0.8 # TODO commented this line, check if including it makes things faster
                    self.step_size *= 0.8
                    #self.step_size = self.step_size * 0.8
                    X, Xval = R.copy(), Rval
                print(R, Rval)
        return R, Rval
    def direction_search(self, X, Xval):
        for i, coordinate in enumerate(self.axes):
            X[coordinate] = X[coordinate]  + self.step_size[i]
            X = self.check_bounds(X)
            self.setup.set_position(X)
            current_point = self.setup.read_input_long(self.num_samples)
            if current_point < Xval: # if new point does not have higher value then go backwards 
                X[coordinate] -= 2 * self.step_size[i]
                X = self.check_bounds(X)
                self.setup.set_position(X)
                current_point = self.setup.read_input_long(self.num_samples)
                if current_point < Xval:
                    X[coordinate] += self.step_size[i]
                else:
                    Xval = current_point
            else:
                Xval = current_point
        if self.setup.mpm is not None:
            self.setup.mpm.set_averaging_time(0.05)
        return X, Xval
    def check_bounds(self, X):
        if any(X[self.axes] <= self.bounds[self.axes, 0]) or any(X[self.axes] >= self.bounds[self.axes, 1]):

            for ax in self.axes:
                if X[ax] <= self.bounds[ax, 0]:
                    X[ax] = self.bounds[ax, 0]
                elif X[ax] >= self.bounds[ax, 1]:
                    X[ax] = self.bounds[ax, 1]

        return X

        
        
    