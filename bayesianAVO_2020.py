"""
Created on Sun Sep 29 02:18:54 2019
@author: Dr. Ayodeji Babalola
"""
#import os.path
import matlab_funcs as mfun
import numpy as np
import Filter as ft
import wellmanager
import AVOModel 
import variogram
import matplotlib.pyplot as plt
import wavelet
import waveletmanager
import SeismicManager
import eigen_matrix_maths as eigenmath
import normal_score_transform as nscore
import randpathV2 as rpath
from progressbar import ProgressBar
import normal_score as normal_score
normal_score = getattr(normal_score,"normal_score")
import NormalScore_Inversion as NormalScore_Inversion
NormalScore_Inversion = getattr( NormalScore_Inversion,"NormalScore_Inversion")
wavelet = getattr(wavelet,"wavelet") # extracting well class
waveletmanager = getattr(waveletmanager,"waveletmanager") # extracting well class
variogram = getattr(variogram,"variogram") 
AVOModel = getattr(AVOModel,"AVOModel") 
wellmanager = getattr(wellmanager,"wellmanager") 
SeismicManager = getattr(SeismicManager,"SeismicManager") 
path = getattr(rpath,"randpath") 
plotd = getattr(mfun,"plotd")
imshow = getattr(mfun,"imshow")
import scipy.linalg as linalg
"""
1.  Seismic Mananger Ver1.... no change in distance arrangement
2.  Neighbourhood is modified so with indices derived from calc_dist
3.  Note in this class, wells are always added
4. Added maximum no of conditioning traces
"""

class bayesianAVO():
    def __init__(self):
        self.sm = SeismicManager()
        self.AVO = None
        self.fhicut = None 
        self.nsamp= None
        self.bkmodfreq= None
        self.fdom = None
        self.vsvp= None
        self.ndat = None
        self.tseis = None
        self.old_dt= None
        self.dt = None
        self.tmax= None
        self.tmin = None
        self.ang = None
        self.nang= None
        self.wavelet= None
        self.snr= None
        self.type = None
        # neighborhood 
        self.vp = None
        self.vs = None
        self.rho = None
        self.VP3D = None
        self.VSP3D = None
        self.RHO3D = None
        self.MS_mean = {}
        self.MS_rand = {}
        self.sim = None
        self.sim_trcs = None
        self.hor= None
        self.HorIndx = None
        self.varvp = None

        self.wellfiles= None
        self.well4amp= None
        self.tlog = None
        self.tcord = None
        self.wellxy= None
        self.logs = None # logs from background
        self.logs_layered = None # resampled to number of layers
        self.wells_loc = {}
        self.wellxy = None
        self.well_indx = None
        self.wellcoord = None
        self.geodata_file = None
      
        #variogram - there are three
        self.vario_vp = variogram()   # import the algorithms
        self.vario_vs = variogram()
        self.vario_rho = variogram()
        self.Cmvp = None
        self.Cmvs = None
        self.Cmrho = None
        self.variances = None
        self.Cm = None
        self.Cd = None # model and data covariance
        self.dobs= None # Data observed
        self.wav_manager = waveletmanager()
        self.well_manager = None
        self.wavelet = None
        self.GA = None
        
        self.wavfile = None # used for loading wavelets
        self.Logs = None
        self.order = None
        self.linear_search = None
        self.EST = None
        self.UNEST = None
        self.sim_indx = None
        self.nsim = None
        self.nwells = None
        self.well_indx = None
        self.MAP = None
        self.Map_var = None
        self.Mu = None
        self.G  = None
        self.MSxy = {}
        self.nlogs = None
        self.LOGS = None
        self.algorithm = None      # Bayesian or Geostat
        self.Geostat_code= None   # Use Matlab or C++ Geostat code  
        self.options = {}

#-------------------------------    
    def init(self):        
        if (self.options['AVO'] is None):
            self.options['AVO'] = 'Aki'
            print('AVO Algorithm is forced to be Aki Equation')
            print('--------------------------------------------')
        self.nCDP = self.sm.nLines * self.sm.nTraces        
        self.nang = len(self.ang)
        self.nwells = len(self.wellfiles)
        #self.linear_search = 10
        self.UNEST = -999
        self.EST = 1
        
        if (self.tmin is None):
            self.tmin = 0
        
        self.tseis = np.arange(self.tmin,self.tmax+self.dt,self.dt)
        if (self.type == "Elastic"):
            self.nlogs  = 4 
        elif(self.type == "Acoustic"):
            self.nlogs =2

        if(hasattr(self.options,'seismic_loaded_memory') == False):
            self.check_size('seismic')
            
        if(hasattr(self.options,'model_loaded_memory') == False):
            self.check_size('models') 

        
        self.vario_vp.init()
        self.vario_vs.init()
        self.vario_rho.init()      
        
        self.vario_vp.setrot2D(0)
        self.vario_vs.setrot2D(0)
        self.vario_rho.setrot2D(0)
        
        # wavelet 
        self.wav_manager.wavefiles = self.wavfile
        self.wav_manager.read()
        self.wavelet = self.wav_manager.get_data()
        #self.wavelet = dnorm.datnorm(self.wavelet)
        # wells
        self.well_manager = wellmanager(self.wellfiles)
        self.well_manager.read()
        self.wellcoord = np.stack((self.well_manager.Inline,self.well_manager.Crossline),axis = 1)
     

        self.logs = mfun.cell(self.nwells,self.nlogs) # This is the resampled log on the seisimic grid 
        self.LOGS = mfun.cell(self.nwells,self.nlogs) # This is the original  log loaded into cells
        self.LOGSZN = mfun.cell(self.nwells,self.nlogs) # This is the original  log loaded into cells
        
        zn = np.zeros((self.nwells,2),dtype=int)
        geodata = mfun.load_obj(self.geodata_file)
        for i in range(self.nwells):
            self.logs[i,0] = geodata['Vplogs'][i,0]
            self.logs[i,1] = geodata['Vplogs'][i,1]
            self.logs[i,2] = geodata['Vslogs'][i,1]
            self.logs[i,3] = geodata['rhologs'][i,1]
            
            # original logs... no time axis
            self.LOGS[i,0] = geodata['VpLOGS'][i,0]
            self.LOGS[i,0] = geodata['VsLOGS'][i,0]
            self.LOGS[i,0] = geodata['rhoLOGS'][i,0]
            self.LOGS[i,1] = geodata['VpLOGS'][i,1]
            self.LOGS[i,2] = geodata['VsLOGS'][i,1]
            self.LOGS[i,3] = geodata['rhoLOGS'][i,1]  
            
        for ii in range(self.nwells):
            tzone_vp = geodata['VpLOGS'][ii,0]
            tzone_vs = geodata['VsLOGS'][ii,0]
            tzone_rho = geodata['rhoLOGS'][ii,0]            
            tzone = np.intersect1d(tzone_vp,np.intersect1d(tzone_vs,tzone_rho))            
            zn[ii,:] = np.array([np.min(tzone),np.max(tzone)])
            self.LOGSZN[ii,1],self.LOGSZN[ii,0] = mfun.segment_logs(zn[ii,:],geodata['VpLOGS'][ii,0],geodata['VpLOGS'][ii,1])
            self.LOGSZN[ii,2],self.LOGSZN[ii,0] = mfun.segment_logs(zn[ii,:],geodata['VsLOGS'][ii,0],geodata['VsLOGS'][ii,1])
            self.LOGSZN[ii,3],self.LOGSZN[ii,0] = mfun.segment_logs(zn[ii,:],geodata['rhoLOGS'][ii,0],geodata['rhoLOGS'][ii,1])            
           
            
        self.init_AVOmodel()
        self.init_SeisManager()
        if (self.options['arb_line']== True):
            self.sm.arb_line()       
 #-------------------------------   
    def init_AVOmodel(self):   
        self.GA = AVOModel()
        self.GA.nsamp = self.nsamp
        self.GA.ang = self.ang
        self.GA.wavelet = self.wavelet
        self.GA.vsvp = self.vsvp
        self.GA.Algo = self.options['AVO']
        #self.GA.process()

#-------------------------------   
    def init_SeisManager(self):

        self.sm.ndat = self.ndat
        self.sm.nsamp = self.nsamp
        self.sm.tmin = self.tmin
        self.sm.tmax = self.tmax
        self.sm.AVO = self.AVO
        self.sm.ang = self.ang
        self.sm.dt = self.dt
        self.sm.old_dt = self.old_dt
        self.sm.varvp = self.varvp
        self.sm.wavelet = self.wavelet
        self.sm.geodata_file = self.geodata_file 
        self.sm.options = self.options
        self.sm.wellcoord = self.wellcoord
        self.sm.vsvp = self.vsvp
        well4sc = {}
         
        indx = 0 
        well4sc['zone']  = self.extract_logs4amp(indx)
        well4sc['vp'] = self.vp
        well4sc['vs'] = self.vs
        well4sc['rho'] = self.rho
        well4sc['tlog'] = self.tlog
        well4sc['indx'] = indx
        self.sm.well4synth = well4sc
        self.sm.nwells = self.nwells
        self.sm.init() # loaded geodata and horizons
        self.hor = self.sm.hor        
        self.wells_loc = self.sm.wells_loc

#-------------------------------    
    def log2seis(self):
        vpb,vsb,rhob = ft.backusaveraging(self.vp,self.vs,self.rho,self.fdom,self.old_dt)
        if(self.old_dt == self.dt):
            self.vp = vpb
            self.vs = vsb
            self.rho = rhob
            self.tcord = self.tlog
        else:
            time1,vpc,vsc,rhoc = mfun.tlcord(self.tlog,self.dt,vpb,vsb,rhob)
            self.rho,self.tcord = ft.resample(rhoc,time1,self.old_dt,self.dt)
            self.vp,self.tcord = ft.resample(vpc,time1,self.old_dt,self.dt)
            self.vs,self.tcord = ft.resample(vsc,time1,self.old_dt,self.dt) 
 
#-------------------------------    
    def CovMat_GI(self,VpCov,VsCov,RhoCov,vp,vs,rho):

        vpstd = mfun.std(vp)
        vsstd = mfun.std(vs)
        rhostd = mfun.std(rho)
        corrvpvs = mfun.corr(vp,vs)
        corrvprho = mfun.corr(vp,rho)
        corrvsrho = mfun.corr(vs,rho)
        
        if (VpCov[0]<0):
            VpCov = np.ones(self.nsamp)*self.variances[0]
        else:
            self.variances[0] = VpCov[0]
        if (VsCov[0]<0):
            VsCov = np.ones(self.nsamp)*self.variances[1]
        else:
            self.variances[1] = VsCov[0]            
        if (RhoCov[0]<0):
            RhoCov = np.ones(self.nsamp)*self.variances[2]
        else:
            self.variances[2] = RhoCov[0]   
            
        CM11 = np.diag(VpCov)
        CM22 = np.diag(VsCov)
        CM33 = np.diag(RhoCov)
        
        CM12 = (vpstd*vsstd*corrvpvs)    *np.eye(self.nsamp)
        CM13 = (vpstd*rhostd*corrvprho)  *np.eye(self.nsamp) 
        CM21 = (vpstd*vsstd*corrvpvs)    *np.eye(self.nsamp)        
        CM23 = (vsstd*rhostd*corrvsrho)  *np.eye(self.nsamp)  
        CM31 = (vpstd*rhostd*corrvprho)  *np.eye(self.nsamp)         
        CM32 = (vsstd*rhostd*corrvsrho)  *np.eye(self.nsamp)   

        self.Cm = np.concatenate((np.concatenate((CM11,CM12,CM13),axis=1) \
                                  ,np.concatenate((CM21,CM22,CM23),axis=1)\
                                  ,np.concatenate((CM31,CM32,CM33),axis=1)),axis=0)  
        
  #-------------------------------    
    def CovMat_GI2(self,VpCov,VsCov,RhoCov,vp,vs,rho):
        vpvar = mfun.var(vp)
        vsvar = mfun.var(vs)
        rhovar = mfun.var(rho)
        vpstd = mfun.std(vp)
        vsstd = mfun.std(vs)
        rhostd = mfun.std(rho)
        corrvpvs = mfun.corr(vp,vs)
        corrvprho = mfun.corr(vp,rho)
        corrvsrho = mfun.corr(vs,rho)
            
        CM11 = (vpvar)    *np.eye(self.nsamp)
        CM22 = (vsvar)    *np.eye(self.nsamp)
        CM33 = (rhovar)    *np.eye(self.nsamp)
        
        CM12 = (vpstd*vsstd*corrvpvs)    *np.eye(self.nsamp)
        CM13 = (vpstd*rhostd*corrvprho)  *np.eye(self.nsamp) 
        CM21 = (vpstd*vsstd*corrvpvs)    *np.eye(self.nsamp)    
        CM23 = (vsstd*rhostd*corrvsrho)  *np.eye(self.nsamp)  
        CM31 = (vpstd*rhostd*corrvprho)  *np.eye(self.nsamp)     
        CM32 = (vsstd*rhostd*corrvsrho)  *np.eye(self.nsamp)   
    
        self.Cm = np.concatenate((np.concatenate((CM11,CM12,CM13),axis=1) \
                                  ,np.concatenate((CM21,CM22,CM23),axis=1)\
                                  ,np.concatenate((CM31,CM32,CM33),axis=1)),axis=0)  
      
 #-------------------------------    
    def CovMat(self,indx = None,flag = None):
        if (flag is None):
            flag = 'geostat'
        elif(indx is None) :
            indx = 0
        indx = np.int(indx)
        vp = np.log(self.sm.VPinterp[indx])
        vs = np.log(self.sm.VSinterp[indx])
        rho = np.log(self.sm.RHOinterp[indx])
        #print ('the background model is giving out np.nan')
        varvp = mfun.var(vp)
        varvs = mfun.var(vs)
        varrho = mfun.var(rho)
        vpstd = mfun.std(vp)
        vsstd = mfun.std(vs)
        rhostd = mfun.std(rho)
        corrvpvs = mfun.corr(vp,vs)
        corrvprho = mfun.corr(vp,rho)
        corrvsrho = mfun.corr(vs,rho)
        self.sm.varvp = vpstd
        if (self.options['modelcovar_type'] == 'scaled_std'):
            CM11 = np.eye(self.nsamp)*(varvp/varvp)
            CM22 = np.eye(self.nsamp)*(varvs/varvp)
            CM33 = np.eye(self.nsamp)*(varrho/varvp)
            
            CM12 = (vpstd*vsstd*corrvpvs/varvp)    *np.eye(self.nsamp)
            CM13 = (vpstd*rhostd*corrvprho/varvp)  *np.eye(self.nsamp) 
            CM21 = (vpstd*vsstd*corrvpvs/varvp)    *np.eye(self.nsamp)            
            CM23 = (vsstd*rhostd*corrvsrho/varvp)  *np.eye(self.nsamp)  
            CM31 = (vpstd*rhostd*corrvprho/varvp)  *np.eye(self.nsamp)             
            CM32 = (vsstd*rhostd*corrvsrho/varvp)  *np.eye(self.nsamp)   

            self.Cm = np.concatenate((np.concatenate((CM11,CM12,CM13),axis=1) \
                                     ,np.concatenate((CM21,CM22,CM23),axis=1)\
                                     ,np.concatenate((CM31,CM32,CM33),axis=1)),axis=0)
        if (self.options['modelcovar_type'] == 'scaled_nostd'):
            CM11 = np.eye(self.nsamp)*(varvp/varvp)
            CM22 = np.eye(self.nsamp)*(varvs/varvp)
            CM33 = np.eye(self.nsamp)*(varrho/varvp)
            
            CM12 = (corrvpvs/varvp)    *np.eye(self.nsamp)
            CM13 = (corrvprho/varvp)  *np.eye(self.nsamp) 
            CM21 = (corrvpvs/varvp)    *np.eye(self.nsamp)            
            CM23 = (corrvsrho/varvp)  *np.eye(self.nsamp)  
            CM31 = (corrvprho/varvp)  *np.eye(self.nsamp)             
            CM32 = (corrvsrho/varvp)  *np.eye(self.nsamp)   

            self.Cm = np.concatenate((np.concatenate((CM11,CM12,CM13),axis=1) \
                                     ,np.concatenate((CM21,CM22,CM23),axis=1)\
                                     ,np.concatenate((CM31,CM32,CM33),axis=1)),axis=0)
        elif (self.options['modelcovar_type'] == 'unscaled_std'):
            CM11 = np.eye(self.nsamp)*(varvp)
            CM22 = np.eye(self.nsamp)*(varvs)
            CM33 = np.eye(self.nsamp)*(varrho)
            
            CM12 = (vpstd*vsstd*corrvpvs)    *np.eye(self.nsamp)
            CM13 = (vpstd*rhostd*corrvprho)  *np.eye(self.nsamp) 
            CM21 = (vpstd*vsstd*corrvpvs)    *np.eye(self.nsamp)            
            CM23 = (vsstd*rhostd*corrvsrho)  *np.eye(self.nsamp)   
            CM31 = (vpstd*rhostd*corrvprho)  *np.eye(self.nsamp)   
            CM32 = (vsstd*rhostd*corrvsrho)  *np.eye(self.nsamp)   

            self.Cm = np.concatenate((np.concatenate((CM11,CM12,CM13),axis=1) \
                                     ,np.concatenate((CM21,CM22,CM23),axis=1)\
                                     ,np.concatenate((CM31,CM32,CM33),axis=1)),axis=0)            
        elif (self.options['modelcovar_type'] == 'unscaled_nostd'):
            CM11 = np.eye(self.nsamp)*(varvp)
            CM22 = np.eye(self.nsamp)*(varvs)
            CM33 = np.eye(self.nsamp)*(varrho)
            
            CM12 = (corrvpvs)   *np.eye(self.nsamp)
            CM13 = (corrvprho)  *np.eye(self.nsamp) 
            CM21 = (corrvpvs)   *np.eye(self.nsamp)
            CM23 = (corrvsrho)  *np.eye(self.nsamp)   
            CM31 = (corrvprho)  *np.eye(self.nsamp)   
            CM32 = (corrvsrho)  *np.eye(self.nsamp)   
            
            self.Cm = np.concatenate((np.concatenate((CM11,CM12,CM13),axis=1) \
                                     ,np.concatenate((CM21,CM22,CM23),axis=1)\
                                     ,np.concatenate((CM31,CM32,CM33),axis=1)),axis=0)     
        else:
            CM11 = np.eye(self.nsamp)*(varvp)
            CM22 = np.eye(self.nsamp)*(varvs)
            CM33 = np.eye(self.nsamp)*(varrho) 
            CM12 = (vpstd*vsstd*corrvpvs)*    np.eye(self.nsamp)
            CM13 = (vpstd*rhostd*corrvprho)*  np.eye(self.nsamp)   
            CM23 = (vsstd*rhostd*corrvsrho)  *np.eye(self.nsamp)
            CM21 = (vpstd*vsstd*corrvpvs)    *np.eye(self.nsamp)       
            CM31 = (vpstd*rhostd*corrvprho)  *np.eye(self.nsamp)   
            CM32 = (vsstd*rhostd*corrvsrho)  *np.eye(self.nsamp) 

            self.Cm = np.concatenate((np.concatenate((CM11,CM12,CM13),axis=1) \
                                     ,np.concatenate((CM21,CM22,CM23),axis=1)\
                                     ,np.concatenate((CM31,CM32,CM33),axis=1)),axis=0)              
          # % I'm simulating spatially-varying covariance matrix
 #---------------------------------
    def extract_logs4amp(self,well_indx):
        tmin = np.zeros(self.nwells,dtype = int)
        tmax = np.zeros(self.nwells,dtype = int)
        indx1 = np.zeros(self.nwells,dtype = int)
        indx2 = np.zeros(self.nwells,dtype = int)        
        
        for i in range(self.nwells):
            tmin[i] = self.logs[i,0][0]
            tmax[i] = self.logs[i,0][-1] # end
            
        tminn = np.amax(tmin)
        tmaxx = np.amin(tmax)
        
        for i in range(self.nwells):
            indx1[i] = mfun.find(self.logs[i,0],tminn)
            indx2[i] = mfun.find(self.logs[i,0],tmaxx)        
     
        """
        self.tlog = self.LOGS[well_indx,0][indx1[well_indx]:indx2[well_indx]] 
        self.vp = self.LOGS[well_indx,1][indx1[well_indx]:indx2[well_indx]] 
        self.vs = self.LOGS[well_indx,2][indx1[well_indx]:indx2[well_indx]] 
        self.rho = self.LOGS[well_indx,3][indx1[well_indx]:indx2[well_indx]] 
        """
        self.tlog = self.logs[well_indx,0][indx1[well_indx]:indx2[well_indx]] 
        self.vp = self.logs[well_indx,1][indx1[well_indx]:indx2[well_indx]] 
        self.vs = self.logs[well_indx,2][indx1[well_indx]:indx2[well_indx]] 
        self.rho = self.logs[well_indx,3][indx1[well_indx]:indx2[well_indx]]
        
        zone = np.array([tmin,tmax])
        return zone[:,well_indx]
 # Note : I want to find the intersection  between the three logs without Nan values  
# Best method is to create average vp,vs and rho logs from resampled logs in creatingsLginGrid fun      
       
#------------------------------------------------------------------------------  
    def randpath(self,ntraces):
        randd = np.random.rand(ntraces)
        indx = np.argsort(randd)
        #indx = np.int(indx)
        return indx
#------------------------------------------------------------------------------  
    def loadwells(self):
        well = mfun.load_obj(self.well4amp)
        vp = well.vp
        vs = well.vs
        rho  = well.vs
        time = well.time
        
        time1,vpc,vsc,rhoc = mfun.tlcord(time,vp,vs,rho)
        old_dt = time1[1] - time1[0]
        vpb1,vsb1,rhob1 = ft.backusaveraging(vpc,vsc,rhoc,self.fdom,old_dt)
        
        if(self.old_dt != self.dt):
            rhob,tcord = ft.resample(rhob1,time1,old_dt,self.dt)
            vp,tcord = ft.resample(vpb1,time1,old_dt,self.dt)
            vs,tcord = ft.resample(vsb1,time1,old_dt,self.dt)
        else:
            rhob = rhob1
            vsb = vsb1
            vpb = vpb1
            tcord = time1
        self.vp = vpb
        self.vs = vsb
        self.rho = rhob
        self.tlog = tcord
            
#------------------------------------------------------------------------------  
    def searchn_v2(self,order_in):
        #self.sim_indx[order_in] = self.EST
        min_neigh = order_in - round(self.linear_search * 0.5)
        max_neigh = order_in + round(self.linear_search * 0.5)
        neigh = np.arange(min_neigh,max_neigh)
        neigh = neigh[neigh>0]
        neigh = neigh[neigh<self.sm.nXL]
        tmp = mfun.find(self.sim_indx[neigh],self.EST)
        indx = neigh[tmp]
        # always adds well-location and remove more than 1
        indx  = np.unique(np.append(indx,self.sm.wells_loc['indx']))       
        return indx.astype('int')
#------------------------------------------------------------------------------  
    def searchn_v3(self,order_in,indx_by_dist): # 
        #1. well indx are always, added
        #2. re-organized index from cal_dist is used to search neighouring traces
        indx = mfun.find(indx_by_dist,order_in)
        min_neigh = indx - round(self.linear_search * 0.5)
        max_neigh = indx + round(self.linear_search * 0.5)
        neigh = np.arange(min_neigh,max_neigh)
        neigh = neigh[neigh>0]
        neigh = neigh[neigh<self.sm.nXL]
        neigh = indx_by_dist[neigh]
        #neigh = neigh[neigh < self.nTraces]
        tmp = mfun.find(self.sim_indx[neigh],self.EST)
        indx = neigh[tmp]
        # always adds well-location and remove more than 1
        indx  = np.unique(np.append(indx,self.sm.wells_loc['indx']))       
        return indx.astype('int')     
#------------------------------------------------------------------------------ 
    def extract_wells(self,neigh_indx):
        if (type(neigh_indx) == int):
            ns = 1
        else:
            ns = neigh_indx.size
        VP = np.zeros((self.nsamp,ns))
        VS = np.zeros((self.nsamp,ns)) 
        RHO = np.zeros((self.nsamp,ns)) 
        
        for i in range(ns):
            if (ns !=1):
                ind = np.int(neigh_indx[i])
            elif(type(neigh_indx) != int and ns==1) :
                ind = neigh_indx[i]            
            else:
                ind = neigh_indx
            VP[:,i] = self.MS_rand['vp'][:,ind]
            VS[:,i] = self.MS_rand['vs'][:,ind]
            RHO[:,i] = self.MS_rand['rho'][:,ind]
        return VP,VS,RHO
#-------------------------------
    def calc_dist(self):
        min_x = np.min(self.sm.seis_coord['X'])
        min_y = np.min(self.sm.seis_coord['Y'])
        dist = np.sqrt(((self.sm.seis_coord['X'] - min_x)**2 + (self.sm.seis_coord['Y'] - min_y)**2))
        indx  = np.argsort(dist)
        return dist,indx.astype(int)   
#------------------------------------------------------------------------------         
    def neigbourhood(self):
        self.MS_rand['vp'] = np.zeros((self.nsamp,self.sm.nXL))
        self.MS_rand['vs'] = np.zeros((self.nsamp,self.sm.nXL))
        self.MS_rand['rho'] = np.zeros((self.nsamp,self.sm.nXL))
        self.MS_mean['vp'] = np.zeros((self.nsamp,self.sm.nXL))
        self.MS_mean['vs'] = np.zeros((self.nsamp,self.sm.nXL))
        self.MS_mean['rho'] = np.zeros((self.nsamp,self.sm.nXL))        
        ind = mfun.gt(self.wells_loc['dist'],10)
        self.wells_loc['dist'] = np.delete(self.wells_loc['dist'],ind)
        self.wells_loc['indx'] = np.delete(self.wells_loc['indx'],ind)
        self.wells_loc['coord'] = np.delete(self.wells_loc['coord'],ind,0)  # last arg is the axis
        self.wells_loc['xy'] = np.delete(self.wells_loc['xy'],ind,0)  
        
        if (self.options['AVO'] == 'Aki'):
            for i in range(len(self.wells_loc['indx'])):
                self.MS_rand['vp'][:,self.wells_loc['indx'][i]] = self.layered[i,0]
                self.MS_rand['vs'][:,self.wells_loc['indx'][i]] = self.layered[i,1]
                self.MS_rand['rho'][:,self.wells_loc['indx'][i]] = self.layered[i,2]
                self.MS_mean['vp'][:,self.wells_loc['indx'][i]] = self.layered[i,0]
                self.MS_mean['vs'][:,self.wells_loc['indx'][i]] = self.layered[i,1]
                self.MS_mean['rho'][:,self.wells_loc['indx'][i]] = self.layered[i,2]                
                self.sim_indx[self.wells_loc['indx'][i]] = self.EST   
        elif(self.options['AVO'] == 'Fatti'):
            for i in range(len(self.wells_loc['indx'])):
                self.MS_rand['vp'][:,self.wells_loc['indx'][i]] = self.layered[i,0]*self.layered[i,2]
                self.MS_rand['vs'][:,self.wells_loc['indx'][i]] = self.layered[i,1]*self.layered[i,2]
                self.MS_rand['rho'][:,self.wells_loc['indx'][i]] = self.layered[i,2]
                self.MS_mean['vp'][:,self.wells_loc['indx'][i]] = self.layered[i,0]*self.layered[i,2]
                self.MS_mean['vs'][:,self.wells_loc['indx'][i]] = self.layered[i,1]*self.layered[i,2]
                self.MS_mean['rho'][:,self.wells_loc['indx'][i]] = self.layered[i,2]                
                #self.sim_indx[self.wells_loc['indx'][i]] = self.EST     
        #self.MS_mean = self.MS_rand
#------------------------------------------------------------------------------ 
    def creatinglogsInGrid(self):    
        Top = self.hor[0]
        Base = self.hor[1]
        
        if (self.options['arb_line'] == True):
            coord = np.vstack((self.sm.bkmcoord['Lines'],self.sm.bkmcoord['Traces']))
            self.sm.wells_loc['indx'],self.sm.wells_loc['dist'] = mfun.kde_tree(self.sm.seis_coord['coord'],self.sm.wells_loc['coord'])
            indx,dist = mfun.kde_tree(coord.T,self.sm.wells_loc['coord']) # index in Horizon
        else:
            coord = Top[:,0:2]
            self.sm.wells_loc['indx'],self.sm.wells_loc['dist'] = mfun.kde_tree(coord,self.sm.wells_loc['coord'])
            indx,dist = mfun.kde_tree(coord,self.sm.wells_loc['coord']) # index in Horizon
        
        self.layered = mfun.cell(self.nwells,self.nlogs)
        for i in range(self.nwells):
            zone  = np.array([Top[indx[i],2], Base[indx[i],2]],dtype=int)
            for ii in range(self.nlogs-1):
                dat,tmp = mfun.segment_logs(zone,self.logs[0,0],self.logs[i,ii+1])
                self.layered[i,ii] = ft.resamp_by_newlength(dat,self.nsamp)
               # self.layered[i,ii] = dat
        
#------------------------------------------------------------------------------ 
    def check_size(self,ttype = None):
        if (ttype == 'seismic'):
            sz = mfun.file_size(self.sm.seisfile[0])            
            if(sz < 1e6):
                self.options['seismic_loaded_memory'] = True
            else:
                self.options['seismic_loaded_memory'] = False
            
        if (ttype == 'models'):
            sz = mfun.file_size(self.sm.VPinterpfile)
                
            if(sz < 1e6):
                self.options['models_loaded_memory'] = True
            else:
                self.options['models_loaded_memory'] = False            
            
#------------------------------------------------------------------------------ 
    def extract_interpModel(self,ind):
        neigh_indx = np.array([ind-1, ind+1])
        neigh_indx = neigh_indx[neigh_indx >0]
        if (type(neigh_indx) == int):
            ns = 1
        else:
            ns = neigh_indx.size
        VP = np.zeros((self.nsamp,ns))
        VS = np.zeros((self.nsamp,ns))
        RHO = np.zeros((self.nsamp,ns))
        for i in range(ns):
            ind = np.int(neigh_indx[i])  # throws an error when used directly.. don't know why
            VP[:,i] = self.sm.VPinterp[ind]
            VS[:,i] = self.sm.VSinterp[ind]
            RHO[:,i] = self.sm.RHOinterp[ind]
        return VP,VS,RHO,neigh_indx
#------------------------------------------------------------------------------ 
    def wellscoord_In_seiscoord(self):  
         FullCoord = np.vstack((self.sm.seis_coord['Lines'] ,self.sm.seis_coord['Traces'] ))
         coord = self.wells_loc['coord']
         indx,dist = mfun.kde_tree(FullCoord.T,coord)  
         self.wells_loc['indx'] = indx
         self.wells_loc['dist'] = dist
#------------------------------------------------------------------------------ 
    def bayesian_aki(self):
        pass

#------------------------------------------------------------------------------ 
    def bayesian_fatti(self):
        # calculates well_indx in the new seismic coordinates arranged by distance
        # bayesian_fatti works better with random path that starts from well-location
        if(self.options['arb_line'] == False):
            self.wellscoord_In_seiscoord()         
        self.neigbourhood()             
        G = self.GA.process()   
        pbar = ProgressBar(maxval= self.sm.nXL).start() 
        self.sim = mfun.cell(self.sm.nXL)
        self.sim_trcs = mfun.cell(self.sm.nXL)
        
        for indx in range(self.sm.nXL):            
            pbar.update(indx)          
            self.sm.Covdatt(indx)
            self.CovMat(indx)
            
            AI_prior = np.log(self.sm.VPmod[indx]*self.sm.RHOmod[indx])
            SI_prior = np.log(self.sm.VSmod[indx]*self.sm.RHOmod[indx])
            RHO_prior = np.log(self.sm.RHOmod[indx])            
           
            Mu = np.hstack((AI_prior,SI_prior,RHO_prior))
            
            geostat_dict = eigenmath.Bayes_aVO(self.Cm,self.sm.Cd,G,Mu,self.sm.dobs)
            mean_ = geostat_dict['mean'].flatten()
            covar_  = geostat_dict['covar']
            
            try :
                R = self.chol(covar_)
            except:
                print ('debugging')
            
            self.MS_mean['vp'][:,indx] = np.exp(mean_[0:self.nsamp])
            self.MS_mean['vs'][:,indx] = np.exp(mean_[self.nsamp:2*self.nsamp])
            self.MS_mean['rho'][:,indx] = np.exp(mean_[2*self.nsamp:3*self.nsamp]) 
            

            
            """
            msg2 = 'neigh_indx ==' + str(neigh_indx)
            msg3 = 'well_indx is ' + str(self.wells_loc['indx'])
            
            print (msg2)
            print(msg3)
            plt.close('all')
                       
            msg1 = 'indx ==' + str(indx)
            print (msg1)
            if(indx==19150):  # test covariance matrix
                break            
             """     
            
        pbar.finish() 
#---------------------------- 
    def realizations(self,mean_,R,G):
        sim = np.zeros((self.nsamp*3,self.nsim))
        sim_trcs = np.zeros((self.nsamp*3,self.nsim))
        corr = np.zeros(self.nsim)
        for i in range(self.nsim):
            sim[:,i] = self.generate_random_samples4(mean_,R)
            sim_trcs[:,i] = G.dot(sim[:,i])
            corr[i] = mfun.corr(sim_trcs[:,i],self.sm.dobs)        
        max_ind = np.argmax(corr)
        best = sim[:,max_ind]        
        return best
#---------------------------- 
    def realizations1(self,mean_,R,G):
        sim = list()
        sim_trcs = list()
        corr = np.zeros(self.nsim)
        for i in range(self.nsim):
            sim.append(self.generate_random_samples4(mean_,R))
            sim_trcs.append(G.dot(sim[i]))
            corr[i] = mfun.corr(sim_trcs[i],self.sm.dobs)        
        max_ind = np.argmax(corr)
        best = sim[max_ind]        
        return best,sim,sim_trcs   
        
#----------------------------      
    def saveModel_AKI(self):
        pass
 
#----------------------------      
    def saveModel_Fatti(self): 
        print ('-------anticipated bugs in savemodel func-> geostat_inv------------------')
        print('1. nhor is forced to be 2 to find the last index')
        print('2. not accouting for the one sample when indexing with [a:b] in numpy')
        self.tmin = np.int(self.sm.bkmcoord['corr_indx']) 
        self.tmax = np.int(np.max(self.hor[1][:,2]))
        self.tseis = np.arange(self.tmin,self.tmax+self.dt,self.dt)     
        nsamp = self.tseis.size
        self.sm.seis_coord['tseis'] = self.tseis
        self.sm.seis_coord['tmin'] = self.tmin
        self.sm.seis_coord['tmax'] = self.tmax
        self.sm.seis_coord['wellcoord'] = self.wells_loc
        self.VP3D_mean  = np.zeros((self.sm.nXL,nsamp))
        self.VS3D_mean  = np.zeros((self.sm.nXL,nsamp))
        self.RHO3D_mean = np.zeros((self.sm.nXL,nsamp))
        extracted_seis  = np.zeros((self.sm.nXL,nsamp,self.nang))
        for i in range(self.sm.nXL):
            Tindex = self.sm.seis_coord['Tindex'][i,:]  # assumed that corr indx is accounted for
            ns = Tindex[1]- Tindex[0] 
            self.VP3D_mean[i,Tindex[0]:Tindex[1]] = ft.resamp_by_newlength(self.MS_mean['vp'][:,i],ns)
            self.VS3D_mean[i,Tindex[0]:Tindex[1]] = ft.resamp_by_newlength(self.MS_mean['vs'][:,i],ns)
            self.RHO3D_mean[i,Tindex[0]:Tindex[1]] = ft.resamp_by_newlength(self.MS_mean['rho'][:,i],ns)
            for ii in range(self.nang):
                extracted_seis[i,Tindex[0]:Tindex[1],ii] = ft.resamp_by_newlength(self.sm.seisdat[i,ii],ns)
        mfun.save_obj('models\\Bayesian\\GeostatModelsMean_Fatti',self.MS_mean) 
        mfun.save_obj('models\\Bayesian\\extracted_seis',extracted_seis)
        mfun.save_obj('models\\Bayesian\\GeostatModelsRand_Fatti',self.MS_rand)
        mfun.save_obj('models\\Bayesian\\seis_coord',self.sm.seis_coord)
        mfun.save_obj('models\\Bayesian\\simulations',self.sim) 
        mfun.save_obj('models\\Bayesian\\simulation_trcs',self.sim_trcs)      
        mfun.numpy_save(self.VP3D_mean,'models\\Bayesian\\AI3D_mean')
        mfun.numpy_save(self.VS3D_mean,'models\\Bayesian\\SI3D_mean')
        mfun.numpy_save(self.RHO3D_mean,'models\\Bayesian\\RHO3D_mean_Fatti')     
        
        print('.........model saved................')       
#----------------------------      
    def nscore_func(self,wlog):  
        zmin = -1.0e21        
        zmax =  1.0e21 
        lt = 1
        ltpar = 0
        ut = 1
        utpar = 0
        """                 
        if (wlog == 'vp'):
            dat = np.log(self.layered.extr_allrows(0))
        elif(wlog == 'vs'):
            dat =np.log(self.layered.extr_allrows(1))       
        elif(wlog == 'rhob'):
            dat = np.log(self.layered.extr_allrows(2))  
        elif(wlog == 'AI'):
            dat = np.log(self.layered.extr_allrows(0) * self.layered.extr_allrows(2)) 
        elif(wlog == 'SI'):
            dat = np.log(self.layered.extr_allrows(1) * self.layered.extr_allrows(2)) 
        """            
        """
        if (wlog == 'vp'):
            dat = np.log(self.logs.extr_allrows(1))
        elif(wlog == 'vs'):
            dat =np.log(self.logs.extr_allrows(2))       
        elif(wlog == 'rhob'):
            dat = np.log(self.logs.extr_allrows(3))  
        elif(wlog == 'AI'):
            dat = np.log(self.logs.extr_allrows(1) * self.logs.extr_allrows(3)) 
        elif(wlog == 'SI'):
            dat = np.log(self.logs.extr_allrows(2) * self.logs.extr_allrows(3))    
        """

        if (wlog == 'vp'):
            dat = np.log(self.LOGSZN.extr_allrows(1))
        elif(wlog == 'vs'):
            dat =np.log(self.LOGSZN.extr_allrows(2))       
        elif(wlog == 'rhob'):
            dat = np.log(self.LOGSZN.extr_allrows(3))  
        elif(wlog == 'AI'):
            dat = np.log(self.LOGSZN.extr_allrows(1) * self.LOGSZN.extr_allrows(3)) 
        elif(wlog == 'SI'):
            dat = np.log(self.LOGSZN.extr_allrows(2) * self.LOGSZN.extr_allrows(3))       

        wt = np.ones(dat.shape)/dat.shape
        ns = nscore.NormalScoreTransform(dat,wt,zmin,zmax,lt,ltpar,ut,utpar) 
        ns.create_transform_func()
        return ns 
 #----------------------------      
    def nscore_func_myversion(self,wlog):
        
        if (wlog == 'vp'):
            dat = np.log(self.layered.extr_allrows(0))
        elif(wlog == 'vs'):
            dat =np.log(self.layered.extr_allrows(1))       
        elif(wlog == 'rhob'):
            dat = np.log(self.layered.extr_allrows(2))  
        elif(wlog == 'AI'):
            dat = np.log(self.layered.extr_allrows(0) * self.layered.extr_allrows(2)) 
        elif(wlog == 'SI'):
            dat = np.log(self.layered.extr_allrows(1) * self.layered.extr_allrows(2)) 
                    
        """
        if (wlog == 'vp'):
            dat = np.log(self.logs.extr_allrows(1))
        elif(wlog == 'vs'):
            dat =np.log(self.logs.extr_allrows(2))       
        elif(wlog == 'rhob'):
            dat = np.log(self.logs.extr_allrows(3))  
        elif(wlog == 'AI'):
            dat = np.log(self.logs.extr_allrows(1) * self.logs.extr_allrows(3)) 
        elif(wlog == 'SI'):
            dat = np.log(self.logs.extr_allrows(2) * self.logs.extr_allrows(3))          
        """
        """
        if (wlog == 'vp'):
            dat = np.log(self.LOGSZN.extr_allrows(1))
        elif(wlog == 'vs'):
            dat =np.log(self.LOGSZN.extr_allrows(2))       
        elif(wlog == 'rhob'):
            dat = np.log(self.LOGSZN.extr_allrows(3))  
        elif(wlog == 'AI'):
            dat = np.log(self.LOGSZN.extr_allrows(1) * self.LOGSZN.extr_allrows(3)) 
        elif(wlog == 'SI'):
            dat = np.log(self.LOGSZN.extr_allrows(2) * self.LOGSZN.extr_allrows(3))       
        """
        ns = normal_score(dat)
        ns.normalscore_transform()
        
        return ns    
#---------------------------------------
    def Bayes_AVO(self,G,Mu): 
        CmG = np.matmul(self.Cm,np.transpose(G))
        GCm = np.matmul(np.transpose(G),self.Cm)
        bb = np.matmul(G,CmG) + self.sm.Cd
        res = self.sm.dobs - G.dot(Mu)
        dd = np.matmul(CmG,np.linalg.pinv(bb))        
        mean_ = Mu +  np.matmul(dd,res)
        covar_ = self.Cm - (np.matmul(dd,GCm))
        #covar_ = 0.5 * (covar_  + covar_.T)  # This takes care of any lack of symmetry in covar..aster
        return mean_,covar_    
    #---------------------------------------
    def generate_random_samples1(self,mean_,covar_): 
        nr,nc = covar_.shape
        invg = np.zeros(nr)
            #rand = np.random.normal(loc=0,scale=1,size = nc)
        rand = np.random.rand(nr)
        for i in range(nr):
            invg[i] = nscore.gauinv(rand[i])
        invg = ft.Filt(invg,self.fhicut,1)
        diag_cov = np.sqrt(np.diag(covar_))
        rand_samp = (diag_cov*invg) + mean_
        return rand_samp
    #---------------------------------------
    def generate_random_samples2(self,mean_,covar_): 
        nr,nc = covar_.shape
        eps = 0.001
        cova = covar_ + np.identity(nr)*eps
        rand_samp  = np.random.multivariate_normal(mean=mean_, cov=cova, size=1)
        return rand_samp.flatten()  
    #---------------------------------------
    def generate_random_samples3(self,mean_,R): 
        nr = mean_.size                          
        rand = ft.Filt(np.random.normal(loc=0,scale=1,size = nr),self.fhicut,1) 
        #rand = np.random.normal(loc=0,scale=1,size = nc)
        rand_samp = np.dot(rand,R) + mean_
        #rand_samp = np.dot(rand,R) + mean_
        return rand_samp  
    #---------------------------------------
    def generate_random_samples4(self,mean_,R): 
        nr = mean_.size
        invg = np.zeros(nr)                          
        rand = np.random.rand(nr)
        for i in range(nr):
            invg[i] = nscore.gauinv(rand[i])
        invg = ft.Filt(invg,self.fhicut,1)
        #rand = np.random.normal(loc=0,scale=1,size = nc) 
        rand_samp = np.dot(invg,R) + mean_
        #rand_samp = np.dot(rand,R) + mean_
        return rand_samp   
    #---------------------------------------
    def chol(self,covar_):
        nr,nc = covar_.shape
        covar_ = covar_*0.5
        eps = 0.00001
        covar_ = covar_ + np.identity(nr)*eps  
        R = linalg.cholesky(covar_)
        return R
#----------------------------    
    def krigging(self,XY_unknown,XY_known,z,wlog,chunksize =None):     
        
        x = XY_known[0,:]
        y = XY_known[1,:]
        xi = XY_unknown[0]
        yi = XY_unknown[1]
        numest = xi.size
        if (chunksize is None):
            chunksize = 200

        if (wlog == 'Vp'):
            Dx = self.vario_vp.bxfun_dist(x,y)
        elif (wlog == 'Vs'):
            Dx = self.vario_vs.bxfun_dist(x,y)    
        elif (wlog == 'Rhob'):
            Dx = self.vario_rho.bxfun_dist(x,y)     
            
        # now calculate the matrix with variogram values 
        if (wlog == 'Vp'):
            At = self.vario_vp.covmat_2D(Dx)
        elif (wlog == 'Vs'):
            At = self.vario_vs.covmat_2D(Dx)
        elif (wlog == 'Rhob'):
            At = self.vario_rho.covmat_2D(Dx)
# the matrix must be expanded by one line and one row to account for
# condition, that all weights must sum to one (lagrange multiplier) 
        nr,nc = At.shape
        A = np.ones((nr+1,nc+1))
        A[0:nr,0:nc] = At  # note 0:3 actually is 0:2, python ugh angry face :)
        A[nr,nr] = 0

#    A is often very badly conditioned. Hence we use the Pseudo-Inverse for
#    solving the equations            
        A = np.linalg.pinv(A)  
        #A = linalg.pinv(A)
        
        z = np.append(z,0) # we also need to expand z
            
# allocate the output zi
        zi = np.empty(numest)
        zi[:]  = np.nan
        s2zi = np.empty(numest)
        s2zi[:] = np.nan
# parametrize engine
        
         # build b
        if (wlog == 'Vp'):
                bx = self.vario_vp.bxfun_dist2(x,xi,y,yi)
        elif (wlog == 'Vs'):
            bx = self.vario_vs.bxfun_dist2(x,xi,y,yi) 
        elif (wlog == 'Rhob'):
            bx = self.vario_rho.bxfun_dist2(x,xi,y,yi)
            
        # now calculate the matrix with variogram values 
        if (wlog == 'Vp'):
            bt = self.vario_vp.covmat_2D(bx)
        elif (wlog == 'Vs'):
            bt = self.vario_vs.covmat_2D(bx)
        elif (wlog == 'Rhob'):
            bt = self.vario_rho.covmat_2D(bx)
         
        nr,nc = bt.shape
        bb = np.ones((nr+1,nc))        
        bb[0:nr,0:nc] = bt  # note 0:3 actually is 0:2, python ugh angry face :)
  
        Lambda = np.matmul(A,bb) 
        zi = np.matmul(Lambda.T,z)        
        s2zi = np.sum(bb*Lambda,0)
        
        return zi,s2zi  
#----------------------------    
    def kriggingV2(self,XY_unknown,XY_known,dat,wlog,chunksize =None):     
        
        x = XY_known[0,:]
        y = XY_known[1,:]
        xi = XY_unknown[0]
        yi = XY_unknown[1]
        numest = xi.size
        if (chunksize is None):
            chunksize = 200

        if (wlog == 'Vp'):
            Dx = self.vario_vp.bxfun_dist(x,y)
        elif (wlog == 'Vs'):
            Dx = self.vario_vs.bxfun_dist(x,y)    
        elif (wlog == 'Rhob'):
            Dx = self.vario_rho.bxfun_dist(x,y)     
            
        # now calculate the matrix with variogram values 
        if (wlog == 'Vp'):
            At = self.vario_vp.covmat_2D(Dx)
        elif (wlog == 'Vs'):
            At = self.vario_vs.covmat_2D(Dx)
        elif (wlog == 'Rhob'):
            At = self.vario_rho.covmat_2D(Dx)
# the matrix must be expanded by one line and one row to account for
# condition, that all weights must sum to one (lagrange multiplier) 
        nr,nc = At.shape
        A = np.ones((nr+1,nc+1))
        A[0:nr,0:nc] = At  # note 0:3 actually is 0:2, python ugh angry face :)
        A[nr,nr] = 0

#    A is often very badly conditioned. Hence we use the Pseudo-Inverse for
#    solving the equations            
        A = np.linalg.pinv(A)  
        #A = linalg.pinv(A)
        
       
            
# allocate the output zi
        zi = np.empty(numest)
        zi[:]  = np.nan
        s2zi = np.empty(numest)
        s2zi[:] = np.nan
# parametrize engine
        
         # build b
        if (wlog == 'Vp'):
                bx = self.vario_vp.bxfun_dist2(x,xi,y,yi)
        elif (wlog == 'Vs'):
            bx = self.vario_vs.bxfun_dist2(x,xi,y,yi) 
        elif (wlog == 'Rhob'):
            bx = self.vario_rho.bxfun_dist2(x,xi,y,yi)
            
        # now calculate the matrix with variogram values 
        if (wlog == 'Vp'):
            bt = self.vario_vp.covmat_2D(bx)
        elif (wlog == 'Vs'):
            bt = self.vario_vs.covmat_2D(bx)
        elif (wlog == 'Rhob'):
            bt = self.vario_rho.covmat_2D(bx)
         
        nr,nc = bt.shape
        bb = np.ones((nr+1,nc))        
        bb[0:nr,0:nc] = bt  # note 0:3 actually is 0:2, python ugh angry face :)
        zi = np.zeros(self.nsamp)
        s2zi = np.zeros(self.nsamp)
        for i in range(self.nsamp):
            Lambda = np.matmul(A,bb)
            z = np.append(dat[i,:],0) # we also need to expand z
            zi[i] = np.matmul(Lambda.T,z)        
            s2zi[i] = np.sum(bb*Lambda,0)
        
        return zi,s2zi  

#----------------------------    
    def kriggingV3(self,XY_unknown,XY_known,vpdat,vsdat,rhodat):     
        # this assumes that spatial covariance is thesame for all
        x = XY_known[0,:]
        y = XY_known[1,:]
        xi = XY_unknown[0]
        yi = XY_unknown[1]
        
        Dx = self.vario_vp.bxfun_dist(x,y)
        At = self.vario_vp.covmat_2D(Dx)
        nr,nc = At.shape
        A = np.ones((nr+1,nc+1))
        A[0:nr,0:nc] = At  # note 0:3 actually is 0:2, python ugh angry face :)
        A[nr,nr] = 0

#    A is often very badly conditioned. Hence we use the Pseudo-Inverse for
#    solving the equations            
        A = np.linalg.pinv(A)  
        #A = linalg.pinv(A)          

# parametrize engine        
         # build b
        bx = self.vario_vp.bxfun_dist2(x,xi,y,yi)
        bt = self.vario_vp.covmat_2D(bx)
         
        nr,nc = bt.shape
        bb = np.ones((nr+1,nc))        
        bb[0:nr,0:nc] = bt  # note 0:3 actually is 0:2, python ugh angry face :)
        vp = np.zeros(self.nsamp)
        cov_vp = np.zeros(self.nsamp)
        vs = np.zeros(self.nsamp)
        cov_vs = np.zeros(self.nsamp)
        rho = np.zeros(self.nsamp)
        cov_rho = np.zeros(self.nsamp)    
        for i in range(self.nsamp):
            Lambda = np.matmul(A,bb)
            z = np.append(vpdat[i,:],0) # we also need to expand z
            vp[i] = np.matmul(Lambda.T,z)        
            cov_vp[i] = np.sum(bb*Lambda,0)

            z = np.append(vsdat[i,:],0) # we also need to expand z
            vs[i] = np.matmul(Lambda.T,z)        
            cov_vs[i] = np.sum(bb*Lambda,0)

            z = np.append(rhodat[i,:],0) # we also need to expand z
            rho[i] = np.matmul(Lambda.T,z)        
            cov_rho[i] = np.sum(bb*Lambda,0)            
        
        return vp,vs,rho,cov_vp,cov_vp,cov_rho    
#---------------------------------------
    def LS(self,G,Mu):
       damp = np.eye(self.nsamp*3)*self.options['damp']        
       aa = np.matmul(G.T,G)+ damp 
       bb = G.T.dot(self.sm.dobs)
       mMap = (eigenmath.ls_qr(aa,bb)).flatten()
       mean_ = Mu + mMap
       return mean_
#---------------------------------------
    def MLE(self,G,Mu): 
        invCd = np.linalg.pinv(self.sm.Cd)
        damp = np.eye(self.nsamp*3)*self.options['damp']
        a1 =np.matmul(G.T,invCd) + damp
        bb = np.matmul(G.T,np.matmul(invCd,self.sm.dobs))
        mMap = (eigenmath.ls_qr(bb,a1)).flatten()
        mean_ = Mu + mMap
        return mean_    
#---------------------------------        
    def QC_plotmodel(self,mean_,Mu,indx,randsamp=None):   
        #plt.close('all')  
        mean_ = np.exp(mean_)
        Mu = np.exp(Mu)
        fig, (ax1, ax2,ax3) = plt.subplots(3,1)
        ns = np.int(mean_.size/3)
        ax1.plot(Mu[0:ns],'k',label= 'BKM')
        if (self.options['AVO'] == 'Aki'):
            ax1.plot(self.sm.VPinterp[indx],'y',label='interp')
        elif(self.options['AVO'] == 'Fatti'):
            ax1.plot(self.sm.VPinterp[indx]*self.sm.RHOinterp[indx],'y',label='interp')
        #ax1.plot(ft.Filt(self.sm.VPinterp[indx],self.fdom,self.dt),'y')
        ax1.plot(mean_[0:ns],'r',label='mean')
        if(randsamp is not None):
             randsamp = np.exp(randsamp)
             ax1.plot(randsamp[0:ns],'b',label=' rand_samples')
        ax1.set_title('Vp')
        ax1.legend() 
        
        ax2.plot(Mu[ns+1:2*ns],'k',label= 'BKM')
        if (self.options['AVO'] == 'Aki'):
            ax2.plot(self.sm.VSinterp[indx],'y',label='interp')
        elif(self.options['AVO'] == 'Fatti'):
            ax2.plot(self.sm.VSinterp[indx] *self.sm.RHOinterp[indx] ,'y') 
        ax2.plot(mean_[ns:2*ns],'r',label='mean')
        if(randsamp is not None):
             ax2.plot(randsamp[ns:2*ns],'b',label=' rand_samples')        
        ax2.set_title('Vs') 
        ax2.legend() 
        
        ax3.plot(Mu[2*ns+1:-1],'k',label= 'BKM')        
        ax3.plot(self.sm.RHOinterp[indx],'y',label='interp')
        ax3.plot(mean_[2*ns:3*ns],'r',label='mean')
        if(randsamp is not None):
             ax3.plot(randsamp[2*ns:3*ns],'b',label=' rand_samples')           
        ax3.set_title('Rhob')
        ax3.legend() 
        
        plt.show()
        plt.pause(1)  
        
#---------------------------------        
    def QC_plotsynth(self,mean_,G): 
        fig, (ax1, ax2,ax3) = plt.subplots(3,1)
        ns = np.int(mean_.size/3)
        synth = G.dot(mean_)
        
        ax1.plot(synth[0:ns],'r')
        ax1.plot(self.sm.dobs[0:ns],'k')
        ax1.set_title('Near')    
        
        ax2.plot(synth[ns:2*ns],'r')
        ax2.plot(self.sm.dobs[ns:2*ns],'k')
        ax1.set_title('Mid') 

        ax3.plot(synth[2*ns:3*ns],'r')
        ax3.plot(self.sm.dobs[2*ns:3*ns],'k')
        ax3.set_title('Far') 
        plt.show()
        plt.pause(1)        
#---------------------------------------
    def process(self):
        self.init()
        INLINE = np.arange(self.sm.INLINE[0],self.sm.INLINE[1],1)            
        if (self.options['arb_line'] == True):
            maxiter = 1
        else:
            maxiter = len(INLINE)            
        self.INLINE = [] # BUG 
        
        print("Geostat inversion started..........")            
        if(self.options['arb_line']== True):
            self.sm.arb_seis_bkm()
        else:
            self.sm.load_seis_bkm()
           
        self.creatinglogsInGrid()
        
        if (self.options['AVO'] == 'Fatti'): # add flag for bayesian AVO when stable
            self.bayesian_fatti()
            self.saveModel_Fatti()
        elif(self.options['AVO'] == 'Aki'): # add flag for bayesian AVO when stable
            self.bayesian_aki()
            self.saveModel_AKI()
        else:
            raise Exception('AVO Algorithm is not defined')
#---------------------------------------            
def edit_log(logdata,num=None):
    if (num is None):
        num = 100    
    ind = mfun.gt(logdata,num) 
    if (ind.size>1):
        for i in range(ind.size):
            logdata[ind[i]] = logdata[ind[i]-1]    
    elif(ind.size ==1):
         logdata[ind] = logdata[ind-1]    
    return logdata
#---------------------------------------            
def edit_logv2(logdata, num=None):
    if (num is None):
        num = 100
    ind = mfun.gt(logdata,num) 
    ind = np.append(ind,mfun.lt(logdata,-num))
    ind = np.append(ind,mfun.find(logdata,np.nan))
    if (ind.size!=0):
        logdata[ind] = np.nan
        logdata = mfun.remove_nan_interp(logdata)
    return logdata   

#---------------------------------------            
def pertub_remove_nan(logdata):
    nr,nc = logdata.shape
    data = np.zeros((nr,nc))
    for i in range(nc):
        #
        grad = np.gradient(logdata[:,i].flatten())
        ind = mfun.find(grad,0) 
        
        if (ind.size!=0):
            pertub = np.arange(ind.size)*0.0001  
            tmp = logdata[:,i].flatten()            
            tmp[ind]= tmp[ind]+ pertub
            data[:,i]  = tmp
        else:
            data[:,i] = logdata[:,i]
    return data 
             
#******************************************************************************            
if __name__ == '__main__':
    plt.close('all')
    SmallDat = bayesianAVO()
    SmallDat.AVO = 2
    SmallDat.fhicut = 1000
    SmallDat.vsvp = 0.5545
    SmallDat.ang = np.array([9, 22, 37])  # Note
    SmallDat.ndat = 1
    
    SmallDat.old_dt = 1
    SmallDat.dt = 1
    SmallDat.bkmodfreq = 20
    SmallDat.fdom = 20  # backus_averaging
    SmallDat.tmin = 2148
    SmallDat.tmax = 2638
    
    SmallDat.GEOM = np.array(([2549766,255916],[5735461,5744011]))
    SmallDat.sm.bin = np.array([30,30])
    SmallDat.sm.rotation = "EW"
    SmallDat.sm.Grid =  np.array(([1450, 1735], [1895, 2000]))
    SmallDat.sm.nLines = 286
    SmallDat.sm.nTraces = 106
    SmallDat.sm.dlines = 1
    SmallDat.sm.dtraces = 1
    SmallDat.sm.horfiles =  np.array(['hor_files\\Quintuco','hor_files\\Tordillo'])   
    SmallDat.wellfiles = np.array(['wells_files\\w1002','wells_files\\w1003'])
    SmallDat.sm.wells_loc['coord'] = np.array(([1483,1927],[1719,1987]))
    SmallDat.sm.type = 'Elastic'
    SmallDat.sm.INLINE = np.array([2850, 3530])
    SmallDat.sm.TRACE = np.array([2200, 2405])
    # files
    SmallDat.sm.seisfile = np.array(['seismic_files\\NearStack','seismic_files\\MidStack','seismic_files\\FarStack' ])
    #SmallDat.sm.seisfile = np.array(['seismic\\FarStack','seismic\\MidStack','seismic\\NearStack' ])
    SmallDat.sm.seiscoordfile = 'seismic_files\\FullStack_coord'
    SmallDat.sm.bkmodcoordfile = 'models\\BKModelcoord'
    SmallDat.wavfile = np.array(['wavelet\\wav_shell_area1'])
    SmallDat.geodata_file = 'models\\geodata'
    
    
    SmallDat.sm.VPmodfile = 'models\\VPmod'
    SmallDat.sm.VSmodfile = 'models\\VSmod'    
    SmallDat.sm.RHOmodfile = 'models\\RHOmod'    
    SmallDat.sm.VPinterpfile = 'models\\VPinterp'     
    SmallDat.sm.VSinterpfile = 'models\\VSinterp'  
    SmallDat.sm.RHOinterpfile = 'models\\RHOinterp'
  
    # variogram
    SmallDat.nsim = 120
    SmallDat.linear_search = 20
    SmallDat.vario_vp.a_hmax = 2*2023
    SmallDat.vario_vp.a_hmin = 2*1678
    SmallDat.vario_vp.a_vert = 1
    SmallDat.vario_vp.c0 = 0
    SmallDat.vario_vp.ang =  np.array([145 ,235, 0])
    SmallDat.vario_vp.nst = 1
    SmallDat.vario_vp.iss = 1
    SmallDat.vario_vp.it = 2
    SmallDat.vario_vp.cc = 1
    
    SmallDat.vario_vs.a_hmax = 2*2023
    SmallDat.vario_vs.a_hmin = 2*1678
    SmallDat.vario_vs.a_vert = 1
    SmallDat.vario_vs.c0 = 0
    SmallDat.vario_vs.ang = np.array([145 ,235, 0])
    SmallDat.vario_vs.nst = 1
    SmallDat.vario_vs.iss = 1
    SmallDat.vario_vs.it = 2
    SmallDat.vario_vs.cc = 1
    
    SmallDat.vario_rho.a_hmax = 2*2023
    SmallDat.vario_rho.a_hmin = 2*1678
    SmallDat.vario_rho.a_vert = 1
    SmallDat.vario_rho.c0 = 0
    SmallDat.vario_rho.ang = np.array([145 ,235, 0])
    SmallDat.vario_rho.nst = 1
    SmallDat.vario_rho.iss = 1
    SmallDat.vario_rho.it = 2
    SmallDat.vario_rho.cc = 1
    
    SmallDat.type = 'Elastic'
    
    # OPTIONS
    SmallDat.options['QC'] = False
    SmallDat.options['datcovar_type'] = 'unscaled'
    SmallDat.options['modelcovar_type'] = 'unscaled_std'
    SmallDat.options['algorithm'] = 'geostatAVO'    
    SmallDat.options['AVO'] = 'Fatti' # Fatti impedance and AKI
    SmallDat.options['std_covar'] = False
    SmallDat.options['std'] = 1e-6
    SmallDat.options['arb_line'] = False
    SmallDat.options['solution'] = 'baye'
    SmallDat.options['damp'] = 1e-4
    
    SmallDat.nsamp = 400 
    SmallDat.process()  


