# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:05:13 2023

@author: s176369
"""

# import sys
# import logging as log
# from time import sleep
# import operator
import sys
sys.path.append("C:\ProgramData\anaconda3\envs\active_alignment\Lib\site-packages\Santec_IL_STS")
import os
# import clr # python for .net
import time
import Santec_IL_STS

# sys.path.append(os.path.dirname(__file__))
from numpy import array
from Santec_IL_STS.mpm_instr_class import MpmDevice
from Santec_IL_STS.tsl_instr_class import TslDevice
from Santec_IL_STS.dev_intr_class import SpuDevice

# ROOT = str(os.path.dirname(__file__))+'\\DLL\\'

# #print(ROOT) #<-- comment in to check if the root was selected properly

# PATH1 ='InstrumentDLL'
# #Add in santec.Instrument.DLL
# ans = clr.AddReference(ROOT+PATH1)

# #print(ans) #<-- comment in to check if the DLL was added properly

from Santec import MPM #　namespace of instrument DLL
from Santec import TSL
from Santec import SPU
from Santec.Communication import CommunicationMethod   # Enumration Class
from Santec.Communication import GPIBConnectType       # Enumration Class

from Santec_IL_STS.error_handing_class import inst_err_str

class Santec_MPM(MpmDevice):
    
    def __init__(self, address = "GPIB0::16::INSTR",  GPIBboard = 0, interface = "GPIB"):
        self.address = int(address.split('::')[1])
        self.GPIBboard = GPIBboard
        self.interface = interface
        self._mpm = MPM()
        self.averaging_time = 0.05


    def set_averaging_time(self, avg_time):
        """Sets the averaging time of the MPM.
        Default is 0.05
        Raises:
            Exception:  In case the MPM is busy;
                        In case of communication failure.
        """
        errorcode = self._mpm.Set_Averaging_Time(avg_time)

        if errorcode != 0:
            raise Exception(str(errorcode) + ": " + inst_err_str(errorcode))
        else:
            self.averaging_time = avg_time
        
    def read_power(self, module_number = 0, channel_number = 1):
        errorcode, power = self._mpm.Get_READ_Power_Channel(module_number, channel_number, 0)
        
        if errorcode != 0:
            raise Exception(str(errorcode) + ": " + inst_err_str(errorcode))

        return power
    
class Santec_TSL(TslDevice):
    
    def __init__(self, address = 4,  GPIBboard = 0, interface = "GPIB"):
        self.address = address
        self.GPIBboard = GPIBboard
        self.interface = interface
        self._tsl= TSL()
        
class Santec_Spu(SpuDevice):
    def __init__(self, devicename = "Dev3"):
        self._spu = SPU()
        self.devicename = devicename
    

