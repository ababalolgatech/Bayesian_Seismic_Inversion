# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 02:46:28 2019

@author: Dr. Ayodeji Babalola
"""
import matlab_funcs as mf
import numpy as np
import forwardmodel as fm

class AVOModel():
    def __init__(self,nsamp = None,ang = None,wav = None,ndat = None,vsvp = None):        
        self.Wavmat  = None 
        self.G = None 
        self.DM  = None
        self.type  = None
        self.Algo  = None
        self.nang = None
        self.init_flag = False
        self.nsamp = None
        self.ang = None
        self.ndat  = None
        self.nsurv = None
        self.wavelet = None
        self.vsvp = None
        self.W = None
        
#------------------    
    def init(self):
        if (self.init_flag is False):
            if (self.ndat is not None):
                self.ndat = 1         
            elif (self.nsamp is  None):
                raise Exception ('Provide nsamp at constructor')
            elif(self.ang is None) :
                raise Exception ('Provide angles at constructor')
            elif(self.vsvp is None) :
                self.vsvp = 0.5
            elif(self.wavelet is None):
                raise Exception ('Provide wavelet file at constructor')
            elif (self.nsurv is None):
                self.nsurv = 1
        """        
        if (self.nang != self.wavelet.size):
            raise Exception ('No of angles is not equal to wavelet') 
        """    
        self.type = 'Elastic'
        self.nang = self.ang.size
        self.nsurv = 1
        if (self.Algo is None):
            self.Algo = 'Aki' 
            
        self.init_flag = True       
 #------------------      
    def process(self):
        if (self.init_flag == False):
            self.init()        
        self.der();
        self.W = self.create_wavdata()
        if (self.Algo == 'Aki'):
            C = self.kern_aki()
        elif(self.Algo == 'Fatti'):
            C = self.kern_fatti()
        elif(self.Algo == 'AI'):
            pass
        else:
            raise Exception ('No Algorithm is specified')
             
        if (self.nang >1):  # pre-stack inversion
            G = np.matmul(self.W,np.matmul(C,self.DM))   
        else: # Post-stack inversion
            G = 0.5*np.matmul(self.W,self.DM) 
        return G
 #------------------   
    def create_wavdata(self):
        if (self.init_flag == False):
            self.init()
        wavz = np.zeros((self.nsamp,self.nsamp))
        wavdata = mf.cell(self.nang*self.nsurv,self.nang*self.nsurv) 
        
        for ii in range(self.nang):
            for i in range(self.nang):
                if (i==ii):
                    wavdata[i,ii] = fm.convmat2(self.wavelet[0,ii],self.nsamp)
                else:
                     wavdata[i,ii]  = wavz
        wavdata = self.cell2mat(wavdata)
        return wavdata.T
  
 #------------------      
    def cell2mat(self,wavdata):
        mat = np.array([])        
        nr,nc = wavdata.size
        for ii in range(nr):
            tmp = []
            for i in range(nc): 
                if (i==0):
                    tmp = wavdata[ii,i]
                else:
                   tmp = np.hstack((tmp,wavdata[ii,i]))
            if(ii == 0):
                mat = tmp
            else:
                mat = np.vstack((mat,tmp))
        return mat         
 #------------------                   
    def kern_aki(self):
         if (self.init_flag == False):
            self.init()            
         t = (self.ang * np.pi) /180
         C1 = mf.cell(self.nang,1)
         C2 = mf.cell(self.nang,1)
         C3 = mf.cell(self.nang,1)
         c1 = np.zeros(self.nsamp)
         c2 = np.zeros(self.nsamp)
         c3 = np.zeros(self.nsamp)  
         
         for i in range (self.nang):
             c1 = np.eye(self.nsamp)* 1/(2*np.cos(t[i]**2))
             c2 = np.eye(self.nsamp)* -4* self.vsvp **2 * (np.sin(t[i]**2))
             c3 = np.eye(self.nsamp)*(0.5* (1+c2))
             
             C1[i,0] = c1
             C2[i,0] = c2
             C3[i,0] = c3
         c1_tmp = self.cell2mat(C1)
         c2_tmp = self.cell2mat(C2)
         c3_tmp = self.cell2mat(C3)
        
         C = np.concatenate((c1_tmp,c2_tmp,c3_tmp),axis = 1)         
         return C 
     
 #------------------                   
    def kern_fatti(self):
         if (self.init_flag == False):
            self.init()            
         t = (self.ang * np.pi) /180
         C1 = mf.cell(self.nang,1)
         C2 = mf.cell(self.nang,1)
         C3 = mf.cell(self.nang,1)
         c1 = np.zeros(self.nsamp)
         c2 = np.zeros(self.nsamp)
         c3 = np.zeros(self.nsamp)  
         
         for i in range (self.nang):
             c1 = np.eye(self.nsamp)* 1/(2*np.cos(t[i]**2))
             c2 = np.eye(self.nsamp)* -4* self.vsvp **2 * (np.sin(t[i]**2))
             c3 = np.eye(self.nsamp)*(0.5 - c2 + (2*self.vsvp**2 * (np.sin(t[i]**2))))
             
             C1[i,0] = c1
             C2[i,0] = c2
             C3[i,0] = c3
         c1_tmp = self.cell2mat(C1)
         c2_tmp = self.cell2mat(C2)
         c3_tmp = self.cell2mat(C3)
        
         C = np.concatenate((c1_tmp,c2_tmp,c3_tmp),axis = 1)         
         return C  
       
 #------------------    
    def der(self):  
        if (self.init_flag == False):
            self.init()        
        ns  = self.nang*self.nsamp
        self.DM = np.zeros((ns,ns))
    
        for i in range(0,ns-1):
            self.DM[i][i] = -1   
            self.DM[i][i+1]= 1   
         
        self.DM[ns-1][ns-1] = -1 

if __name__ == '__main__':
    import waveletmanager
    waveletmanager = getattr(waveletmanager,"waveletmanager") # extracting well class
    """
    G = AVOModel()
    G.nsamp = 10
    G.ang = np.array([10,20,30])
    wav_manager = waveletmanager()
    wav_manager.wavefiles = np.array(['wavelet\\wav_poseidon'])
    wav_manager.read()
    wavelet = wav_manager.get_data() 
    
    G = AVOModel()
    G.nsamp = 10
    G.ang = np.array([10,20,30])
    G.wavelet = wavelet
    G.process()
    """
    G = AVOModel()
    wav_manager = waveletmanager()
    wav_manager.wavefiles = np.array(['wavelet\\wav_shell_area1_Full'])
    wav_manager.read()
    #wavelet = wav_manager.get_data() 
    dat = np.array([1,2,3])
    wavelet  = mf.cell(1,1)
    wavelet[0,0] = dat
    
    
    G = AVOModel()
    G.nsamp = 5
    G.ang = np.array([0])
    G.wavelet = wavelet
    GG = G.process()
    
    

     
             
             
             
             
        
        

            
        
            
        
        
        