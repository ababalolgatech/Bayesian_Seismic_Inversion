# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:58:16 2019

@author: Dr. Ayodeji Babalola
"""
import matlab_funcs as mfun
import numpy as np
import Filter as ft
from scipy import signal
from scipy.fftpack import fft, ifft
plotd = getattr(mfun,"plotd") 
class wavelet:
    def __init__(self):
       self.fdom = None
       self.tmax = None
       self.tmin = None
       self.data = None
       self.nang = None
       self.nsamp = None
       self.Type = None
       self.seismic_traces = None
       self.tseis = None
       self.seis_freq = None
       self.method = None
       self.wavename  = None
       self.ampsec = None
       self.freq = None
       self.init_flag = None

    def __repr__(self):
        if (self.wavename is None):
            self.wavename = 'wavelet class'
        return repr(self.wavename) 
    def  init(self):
        if (self.init_flag is None):
            if (self.data is not None):
                self.nsamp,self.ang = self.data.shape
                self.type = 'extract'
                self.init_flag = True
#----------------------             
    def saveObj(self):
        if (self.data is not None):
            fname = self.wavename
            mfun.save_obj(fname,self)    
#----------------------             
    def ricker(self):
         pass
#----------------------      
    def ormsby(self) :
         pass
    def statistical_extraction1 (self,trc = None,dt= None,nsamp= None,Type= None):
        trct = trc
        trcorr,maxlag = ft.xcorr(trct)
        tmp_indx = np.int(nsamp/2)
        indx = np.arange(maxlag-tmp_indx-1,maxlag+tmp_indx )
        trcorn = trcorr[indx]
        trt2 = signal.hanning(trcorn.size)
        trcorn = trt2*trcorn
        fData = fft(trcorn)
        fdat = np.sqrt(np.abs(fData))
        
        # Establish the number of samples of the output dataset    
        nsamp = fdat.size
        if (np.mod(nsamp,2) ==0): # even
            nsamph = np.int(nsamp/2)
        else:
            nsamph = np.int((nsamp+1)/2)
        
        # remove zeros in the spectrum
        llimit = np.max(fdat)*1.0e-4 
        fdat[fdat<llimit] = llimit
        
        temp = fdat.astype('complex')
        temp = np.real(ifft(temp))
        end_ = temp.size
        wav = np.concatenate((temp[nsamph-1:end_],temp[0:nsamph-1]))

        wav = self.remove_dc(wav)
        wav = wav/np.max(np.abs(wav))
        
        tmax = (nsamp-1)*dt
        t = np.linspace(-tmax*0.5,tmax*0.5,nsamp)
        
        # Apply taper to the final wavelet
        trt = signal.hanning(wav.size)
        wav = wav*trt
        
        return wav,t
    
    def statistical_extraction2 (self,trc = None,dt= None,nsamp= None,Type= None):
        nt_wav = np.int(nsamp/2) # lenght of wavelet in samples
        nfft = 2**11 # lenght of fft
        
        # time axis for wavelet
        t_wav = np.arange(nt_wav) * (dt/1000) 
        t_wav = np.concatenate((np.flipud(-t_wav[1:]), t_wav), axis=0)    
        trt = signal.hanning(trc.size)
        trc = trt*trc # apply taper to the trace
        # estimate wavelet spectrum       
        wav_est_fft = np.abs(np.fft.fft(trc,nfft))
        fwest = np.fft.fftfreq(nfft, d=dt/1000)
        
        # create wavelet in time
        wav = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
        wav = np.concatenate((np.flipud(wav), wav), axis=0)
        wav = wav / wav.max()
        wcenter = np.argmax(np.abs(wav))
        trt = signal.hanning(wav.size)
        wav = trt*wav        
        wav = self.remove_dc(wav)
        # Apply taper to the final wavelet
        trt = signal.hanning(wav.size)
        wav = wav*trt

        return wav, fwest    
#----------------------    
    def least_squares(self,percent):
         pass  
#----------------------     
    def remove_dc(self,wav):        
        posIndx = mfun.gt(wav,0)
        negIndx = mfun.lt(wav,0)
        pos = wav[posIndx]
        neg = wav[negIndx]
        sc = np.abs(np.sum(pos))/np.abs(np.sum(neg))     
        wav[negIndx] = sc*wav[negIndx]
        return wav
#----------------------    
    def apply_taper(self):
        pass
#----------------------    
    def cal_spectrum(self,wav,dt):
        fdat = np.sqrt(np.abs(fft(wav))) # fft might require some zero padding
        nt = wav.size
        tmax = (nt-1)*dt               
        df = 1/0.001*tmax
        nf = nt
        fmax = (nf-1)*df
        fmin = fmax-(nf-1)*df
        f = np.arange(fmin,fmax,df)
        
        return fdat, f    
    
 #----------------------------------------------- 
if __name__ == "__main__": 
    import scipy.io as io
    import matplotlib.pyplot as plt
    tr = io.loadmat('trc_test.mat')
    trc = tr['trc_test'].flatten()
    wm = wavelet()
    dt = 1
    wav,t = wm.statistical_extraction(trc,dt,121)
    plt.plot(wav,t)
    
    