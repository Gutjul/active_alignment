# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:27:37 2023

@author: miju
"""
from TLPM import TLPM
from ctypes import CDLL,cdll,c_long, c_ulong, c_uint32,byref,create_string_buffer,c_bool,c_char_p,c_int,c_int16,c_double, sizeof, c_voidp, c_short

class Powermeter():
    def __init__(self):
        # initialize Powermeter               
        self.tlPM = TLPM()
        self.deviceCount = c_uint32()
        self.isPowermeter = True
        
        
    def GetAddress(self):
        self.tlPM.findRsrc(byref(self.deviceCount))
        self.resourceName = create_string_buffer(1024)
        self.tlPM.getRsrcName(c_int(0), self.resourceName)
        self.address = c_char_p(self.resourceName.raw).value
        return self.address # Return the address
        
        
        
    def Connect(self, resourceNameValue):
        try:
            self.tlPM.findRsrc(byref(self.deviceCount))
            self.resourceName = create_string_buffer(1024)
            for i in range(0, self.deviceCount.value):
                self.tlPM.getRsrcName(c_int(i), self.resourceName)
                print(c_char_p(self.resourceName.raw).value)
                if resourceNameValue == c_char_p(self.resourceName.raw).value:
                    break
            self.tlPM.close()
            self.tlPM = TLPM()
            self.tlPM.open(self.resourceName, c_bool(True), c_bool(True))
        except:
            print("Power meter not plugged in")
            self.isPowermeter = False
        
        
    def SetWl(self,wl):
        # self.tlPM = TLPM()
        # self.tlPM.open(self.resourceName, c_bool(True), c_bool(True))
        wlset =  c_double(wl)
        self.tlPM.setWavelength(wlset)    
        # self.tlPM.close()
    def SetWlandMeasPower(self,wl):
        # self.tlPM = TLPM()
        # self.tlPM.open(self.resourceName, c_bool(True), c_bool(True))
        wlset =  c_double(wl)
        self.tlPM.setWavelength(wlset)
        self.power =  c_double()
        self.tlPM.measPower(byref(self.power))
        # self.tlPM.close()
    def MeasurePower(self):
        # self.tlPM = TLPM()
        # self.tlPM.open(self.resourceName, c_bool(True), c_bool(True))
        self.power =  c_double()
        self.tlPM.measPower(byref(self.power))
        # self.tlPM.close()
    def GetAvgTime(self):
        """Averaging time given in seconds"""
        self.AvgTime = c_double()
        self.Attribute = c_int16(0) # 0 for set value
        self.tlPM.getAvgTime(self.Attribute, byref(self.AvgTime))
        self.MinAvgTime = c_double()
        self.Attribute = c_int16(1) # 1 for minimum value
        self.tlPM.getAvgTime(self.Attribute, byref(self.MinAvgTime))
        self.MaxAvgTime = c_double()
        self.Attribute = c_int16(2) # 2 for maximum value
        self.tlPM.getAvgTime(self.Attribute, byref(self.MaxAvgTime))
        self.DefAvgTime = c_double()
        self.Attribute = c_int16(3) # 3 for default value
        self.tlPM.getAvgTime(self.Attribute, byref(self.DefAvgTime))
        
        
        
    def SetAvgTime(self, AvgTime):
        """Averaging time given in seconds
            Default value: 0.001 s
            Minimum value: 0.001 s
            Maximum value: 32.767 s
        """
        self.AvgTime = c_double(AvgTime)
        self.tlPM.setAvgTime(self.AvgTime)
        
    
    def GetAvgCount(self):
        self.AvgCount = c_int16()
        self.tlPM.getAvgCnt(byref(self.AvgCount))
        
    def SetAvgCount(self, AvgCount):
        self.AvgCount = c_int16(AvgCount)
        self.tlPM.setAvgCnt(self.AvgCount)
    
    def GetPowerRange(self):
        """ Gives the power range in watts"""
        Attribute = c_int16(0) # 0 for set value
        self.powerValue = c_double()
        self.tlPM.getPowerRange(Attribute, byref(self.powerValue))
        Attribute = c_int16(1) # 1 for minimum value
        self.minPower = c_double()
        self.tlPM.getPowerRange(Attribute, byref(self.minPower))
        Attribute = c_int16(2) # 2 for maximum value
        self.maxPower = c_double()
        self.tlPM.getPowerRange(Attribute, byref(self.maxPower))
        
    def read_input(self):
        self.MeasurePower()
        return self.power.value
    
    def read_input_long_continous(self, num_samples = 50):
        # PM101 has a readout rate of 1 kilo samples per second
        self.SetAvgTime(num_samples/1000)
        self.MeasurePower()
        self.SetAvgTime(0.001)
        return self.power.value
    
    def ClosePwrMtr(self):
        self.tlPM.close()
            
#PwrMtr = Powermeter("hej")
# b'USB0::0x1313::0x8076::M00977406::INSTR'