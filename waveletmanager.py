# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:50:44 2019

@author: Ayodeji Babalola
"""

import numpy as np
import matlab_funcs as mfun   
import wavelet as wav
wavelet = getattr(wav,"wavelet") # extracting well class

class waveletmanager:
    def __init__(self):
        self.wav = None
        self.nsamp = None
        self.wavefiles = None
        self.wavename = None
        self.nwav = None
        self.avg_wavlet = None
        self.seis_data = None
        self.init_flag = None
        self.wavmat = None
        self.data = None
        self.nang = None
        self.type = None
        
#---------------------------------------         
    def __repr__(self):
        if (self.wavename is None):
            self.wavename = 'waveletManger class'
        return repr(self.wavename)         
#---------------------------------------          
    def init (self):
        if (self.wavefiles is None and self.type is None ):
            if (self.wavefiles is None):
                raise Exception ("Please add wavelet file names")
            elif(self.type is None):
                raise Exception ("Are you calculating mathematical wavelet")
        if (self.wavefiles is not None): 
            if (type(self.wavefiles) is not np.ndarray):
                raise Exception ('The wavelet files must Numpy Arrays')
            else:
                self.nwav = len(self.wavefiles)
        else:
              self.nwav = 1
        
        self.wav = mfun.create_obj(wavelet,self.nwav)
        
        self.wavename = np.chararray(self.nwav)
        self.init_flag = True      
#---------------------------------------      
    def read(self):
       if (self.init_flag is None):
           self.init()  
       
       tmp = mfun.load_obj(self.wavefiles[0]) 
       self.data = mfun.cell(self.nwav,tmp.nang)
       for ii in range(self.nwav):
           self.wav[ii] = mfun.load_obj(self.wavefiles[ii]) 
           self.nang = self.wav[ii].nang
           self.wavename[ii] = self.wav[ii].wavename           
           
           for i in range(self.nang):
               if (self.nang == 1):
                   self.data[ii,i] = self.wav[ii].data
               else:
                    self.data[ii,i] = self.wav[ii].data[:,i]
               
#---------------------------------------          
    def get_data(self):
        if (self.init_flag is None):
            self.init()
        if(self.wav is None):
            self.read()

        #for i in range(self.nwav):
            #self.wav[i].remove_dc() ;
            #self.wav[i].normalize();
        return self.data
#---------------------------------------  
    def cellreorganize():
        # remove cell inside a cell        
        pass     
    
        
    