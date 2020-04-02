# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 02:40:59 2019
@author: Dr. Ayodeji Babalola
"""
import numpy as np
#from numba import jit
class variogram:
    def __init__(self):
        self.a_hmax= None 
        self.Type = None
        self.a_hmin = None
        self.a_vert= None
        self.anis1 = None  
        self.anis2 = None 
        self.iss = None
        self.c0 = None
        self.cc = None
        self.it = None
        self.aa = None 
        self.nst= None 
        self.ang = None # ang = [ang1 ang2 ang3]
        self.rotmat = None
        self.MAXNST = None
        self.MAXROT = None
        self. CovMat = None
        self.data= None
        self.covmtx = None
        self.EPSLON = None
        self.spatial_type = None

#---------------------------- 
    def __repr__(self):
        return repr("Variogram Class")        
    
    
#---------------------------- 
    def init(self,spatial_type = None):
        self.spatial_type = spatial_type 
        if (self.spatial_type is None):
            self.spatial_type = '2D'
        self.MAXNST = 4
        self.MAXROT = self.MAXNST + 1
        self.EPSLON = 1e-16         
        
        if (self.spatial_type == '3D'):
            self.rotmat = np.zeros(((self.iss+1,3,3)))
            self.anis1 = self.a_hmin/self.a_hmax
            self.anis2 = self.a_vert/self.a_hmax    
        else:
            self.rotmat = np.zeros((4,self.nst))
            self.anis = self.a_hmin/self.a_hmax
 
#----------------------------
    def covm_sgs_ed(self,data,GEOM):      
        if (self.rotmat is None):
            self.init()
        
        x = data['x']
        y = data['y']
        z = data['z']
        
        ns = z.size
        x1 = GEOM[0,0]
        y1 = GEOM[1,0]
        z1 = z[0]
        
        covmtx = np.zeros((ns,ns))
        
        for i in range(ns):
            covmtx[i,i] = self.cova3_ed(x1,y1,z1,x,y,z[i],1,1)
            
        return covmtx
#----------------------------
    def setrot(self,ind):  
        if (self.spatial_type == '2D'):
            self.setrot2D(ind)
        elif(self.spatial_type == '3D'):
            self.setrot3D(ind)
        
#----------------------------
    def setrot3D(self,ind):
        # Setting up rotation matrices for variogram and search
        DEG2RAD = 3.141592654/180.0 
        EPSLON  = 1.e-20
        
        if (self.ang[0] >= 0.0  and self.ang[0] < 270):
            alpha = (90.0   - self.ang[0]) * DEG2RAD 
        else:
            alpha = (450.0  - self.ang[0]) * DEG2RAD ;
         
        beta = -1.0 * self.ang[1] * DEG2RAD  
        theta =  self.ang[2] * DEG2RAD 
        
        sina  = np.sin(alpha)  
        sinb  = np.sin(beta)   
        sint  = np.sin(theta)  
        cosa  = np.cos(alpha)  
        cosb  = np.cos(beta)   
        cost  = np.cos(theta)
        
        afac1 = 1.0 / max(self.anis1,EPSLON) 
        afac2 = 1.0 / max(self.anis2,EPSLON) 
        self.rotmat[ind,0,0] =           (cosb * cosa)   
        self.rotmat[ind,0,1] =             (cosb * sina)   
        self.rotmat[ind,0,2] =       (-sinb)         
        self.rotmat[ind,1,0] = afac1*(-cost*sina + sint*sinb*cosa) 
        self.rotmat[ind,1,1] = afac1*(cost*cosa + sint*sinb*sina)  
        self.rotmat[ind,1,2] = afac1*( sint * cosb)                
        self.rotmat[ind,2,0] = afac2*(sint*sina + cost*sinb*cosa)  
        self.rotmat[ind,2,1] = afac2*(-sint*cosa + cost*sinb*sina) 
        self.rotmat[ind,2,2] = afac2*(cost * cosb)       

#----------------------------
    def setrot2D(self,ind):
        # Setting up rotation matrices for variogram and search
        azumth = np.deg2rad(90.0 - np.array(self.ang[0]))
        self.rotmat = np.zeros((4, self.nst))
        self.rotmat[0] = np.cos(azumth)
        self.rotmat[1] = np.sin(azumth)
        self.rotmat[2] = -np.sin(azumth)
        self.rotmat[3] = np.cos(azumth)
        
#----------------------------
    def cova3_ed(self,x1,y1,z1,x2,y2,z2,ivarg,irot):
    #     ivarg  :variogram number (set to 1 unless doing cokriging or indicator kriging)   
    #    irot  :           index of the rotation matrix for the first nested   structure 
    #   (the second nested structure will use irot+1, the third irot+2, and so on)
        EPSLON  = 1.e-20
        istart  = 1 + (ivarg-1)*self.MAXNST
        cmax = self.c0
        
        for i in range(self.nst):
            cmax = cmax + self.cc

    # check for zero distance
        hsqd = self.sqdist3D(x1,y1,z1,x2,y2,z2,irot)
        """
        if (np.real(hsqd) < EPSLON):
            cova = cmax
            return
        """   
        if (np.real(hsqd)==0 or np.real(hsqd) < EPSLON):
            cova = cmax
            return  cova      
    # Loop over all the structures:
        cova = 0.0    
  
            
        hsqd = self.sqdist3D(x1,y1,z1,x2,y2,z2,0)            
        h = np.real(np.sqrt(hsqd))
            
            # Spherical Variogram Model
        if (self.it == 1):
            hr = h/self.a_hmax
            if (hr < 1):
                cova  = cova + self.cc * (1-hr*(1.5 - 1.5*hr*hr))
            # Exponential Variogram Model            
        elif(self.it == 2) :
            cova  = cova + self.cc * np.exp(-3.0*h/self.a_hmax)
           # Gaussian Variogram Model     
        elif(self.it == 3):
               cova = cova + self.cc*np.exp(-3.*(h/self.a_hmax)*(h/self.a_hmax)) 
          #  Power Variogram Model   
        elif(self.it == 4):  
             cova = cova + cmax - self.cc*(h**self.a_hmax)              
          # Hole Effect Model       
        elif(self.it == 5):
             d = 10.0 * self.a_hmax
             cova = cova + self.cc*np.exp(-3.0*h/d)*np.cos(h/self.a_hmax*np.pi) ;
             cova = cova + self.cc*np.cos(h/self.a_hmax*np.pi)
             
        return cova
#----------------------------
    def sqdist3D(self,x1,y1,z1,x2,y2,z2,ind):
        dx = np.double(x1 - x2)
        dy = np.double(y1 - y2)
        dz = np.double(z1 - z2)
        dist = []
        for i in range(3):
            cont = self.rotmat[ind,i,0]* dx \
            + self.rotmat[ind,i,1]* dy \
            + self.rotmat[ind,i,2]* dz 
            dist = dist + cont**2
            
        if(dist.size == 0):
            dist = 0
        return dist

 #----------------------------
    #@jit(nopython=True)
    def sqdist2D(self,x1,y1,x2,y2,ind):
        dx = x1 - x2
        dy = y1 - y2
        dist = []
        dx1 =  dx*self.rotmat[0] + dy*self.rotmat[1]
        dy1 = (dx*self.rotmat[2] + dy*self.rotmat[3])/self.anis
        dist = dx1*dx1 + dy1*dy1
        return dist        
  #----------------------------
    def bxfun_dist(self,x,y):       
        nx = x.size    
        ny = y.size # nx should be equal to ny.. mimicking the matlab_funcs version   
        tmp_x = np.tile(x,(nx,1))
        tmp_x = tmp_x.T
        tmp_y = np.tile(y,(ny,1))
        tmp_y = tmp_y.T        
        dist = np.zeros((nx,ny))
        
        for ii in range(ny):
            for i in range(nx):
                dist[i,ii] = self.sqdist2D(tmp_x[i,ii],tmp_y[i,ii],x[ii],y[ii],0) 
                # should always be zero unless doing cokriging or indicator kriging    
        return dist 
    
   # @jit(nopython=True)
  #----------------------------
    def bxfun_dist2(self,x1,x2,y1,y2):       
        n2 = x2.size 
        if (n2==1):
            x2 = np.array([x2])
            y2 = np.array([y2])
        n1 = x1.size # nx should be equal to ny.. mimicking the matlab_funcs version   
        if (n1==1):
            x1 = np.array([x1])
            y1 = np.array([y1])        
        
        tmp_x = np.tile(x1,(n2,1))
        tmp_x = tmp_x
        tmp_y = np.tile(y1,(n2,1))
        #tmp_y = tmp_y       
        dist = np.zeros((n2,n1))
        
        for ii in range(n1):
            for i in range(n2):
                dist[i,ii] = self.sqdist2D(tmp_x[i,ii],tmp_y[i,ii],x2[i],y2[i],0) 
                # should always be zero unless doing cokriging or indicator kriging
        return dist.T 
  #----------------------------
    #@jit(nopython=True)
    def covmat_2D(self,hh): 
      #EPSLON = 2.220446049250313e-16
    # h is distance calculated from bxfun_dist
      #nr,nc = Dx.shape
      #covmat = np.zeros((nr,nc))
      
      """
      nr,nc = hh.shape
      if (nr==1 and nc==1 and hh[nr] < EPSLON):          
          cova = np.ones((nr,nc))*self.cc
          return cova 
      """
      hh = np.real(np.sqrt(hh)) 
      if (self.Type is None):
            self.Type = 'bounded'
      if (self.Type == 'bounded'):
            hh[hh>self.a_hmax] = self.a_hmax  
            
      if (self.it == 1):
            hr = hh/self.a_hmax
            if (hr < 1):
                cova  = self.cc * (1-hr*(1.5 - 1.5*hr*hr))
            # Exponential Variogram Model            
      elif(self.it == 2) :
                cova  = self.cc * np.exp(-3.0*hh/self.a_hmax)
           # Gaussian Variogram Model     
      elif(self.it == 3):
               cova = self.cc*np.exp(-3.*(hh/self.a_hmax)*(hh/self.a_hmax))
                
      return cova

 
    
        
        
#******************************************************************************            
if __name__ == '__main__':            
    # variogram
    vario = variogram()
    vario.a_hmax = 512
    vario.a_hmin = 200
    vario.a_vert = 2
    vario.c0 = 0
    vario.ang = np.array([13,103,90])
    vario.nst = 1
    vario.iss = 1
    vario.it = 2
    vario.cc = 0.16 
    vario.init('2D')
    vario.setrot2D(0)

    
    """
    data = {}         
    GEOM = np.array(([422811.47, 431621.93],[8487174.21 ,8497673.86]))
    data['x'] = np.linspace(422811.47,431621.93,10)
    data['y'] = np.linspace(8487174.21,8497673.86,10)
    data['z'] = np.linspace(10,100,10)
    Cmvp = vario.covm_sgs_ed(data,GEOM)  
    """
    x = np.array([1000,2100])
    y = np.array([5000,6100])
    x1 = np.array([1000,1500,2100,3000])
    y1 = np.array([5000,5500,6100,9000])
    print(np.sqrt(vario.bxfun_dist(x,y)))
    vario.bxfun_dist2(x,x1,y,y1)
    vario.bxfun_dist2(x,y,x,y)
    # Not properly-implimented... check matlab krigging code          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                       
      
      
      
        
        