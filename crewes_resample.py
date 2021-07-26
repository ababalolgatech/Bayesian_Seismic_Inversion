# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:06:23 2019

@author: Dr. Ayodeji Babalola
"""
import numpy as np
from scipy.linalg import toeplitz
import matlab_funcs as mfun
import eigen_matrix_maths as eigenmath
#from eigen_matrix_maths import*

#-----------------------------------------------------------------
def resamp_crewes(trin,t,dtout,timesout = None,flag=None,fparams = None):
    if (flag is None):
        flag = 1
    if(fparams is None):
        fparams = np.array([.6,120])
    if(timesout is None):
        timesout = np.array([t[0],t[-1]])
        
    dtin = t[1]-t[0]
    
    # build output time vector
    tout = np.arange(timesout[0],timesout[1]+dtout,dtout)
    
    ilive  = mfun.findnot(trin,np.nan)
   # check for completely dead trace
    if (len(ilive) == 0):
       trout = np.nan*np.ones(tout.size)
       return
   
    if (flag == 1):
        norder = 8 
    else:
        norder  = 4    
    if(dtout > dtin):
        raise Exception ("Undersampling is not yet Implimented")
    
    trout = sincnan(trin,t,tout)
    
    return trout,tout
    
#-----------------------------------------------------------------
def between(x1,x2,testpts,flag = None):
    # logical test, finds samples in vector between given bounds
    if (flag is None):
        flag = 0
    if(x1.size !=1 or x2.size !=1):
        raise Exception ('x1 and x2 must be scalars')
    
    if (flag == 0):
        if (x1<x2):
            indices = np.intersect1d(np.argwhere(testpts > x1),np.argwhere(testpts < x2))
            if (indices.size==0):
                indices = 0
                return
        else:
            indices = np.intersect1d(np.argwhere(testpts > x2),np.argwhere(testpts < x1))
            if (indices.size==0):
                indices = 0
                return
            
    if (flag == 1):
        if (x1<x2):
            indices = np.intersect1d(np.argwhere(testpts >= x1),np.argwhere(testpts < x2))
            if (indices.size==0):
                indices = 0
                return
        else:
            indices = np.intersect1d(np.argwhere(testpts > x2),np.argwhere(testpts <= x1))
            if (indices.size==0):
                indices = 0
                return
    
    if (flag == 2):
        if (x1<x2):
            indices = np.intersect1d(np.argwhere(testpts >= x1),np.argwhere(testpts <= x2))
            if (indices.size==0):
                indices = 0
                return
        else:
            indices = np.intersect1d(np.argwhere(testpts >= x2),np.argwhere(testpts <= x1))
            if (indices.size==0):
                indices = 0
                return
        return indices
    
#-----------------------------------------------------------------    
def sincnan(trin,t,tout,sizetable = None):
   """
 SINCINAN: sinc function interpolation for signals with embedded nan's 

  trout=sincinan(trin,t,tout,sizetable)
  trout=sincinan(trin,t,tout)
 
 SINCINAN performs 8 point sinc function interpolation using a 
 design for the approximate sinc function due to Dave Hale. It
 differs from SINCI in that it is specially modified to deal with
 the presence of NaN's in a sensible manner. SINCI will cause any
 trace portions containing NaN's to grow larger by the length of
 the interpolation function. SINCINAN avoids this by breaking the 
 input trace into live segments and interpolating each separatly.
 Each segment gets the same constant extrapolation end treatment as
 the trace as a whole; however this only affects interpolation sites
 falling within the live segment. Interpolation sites falling in NaN
 zones or outside the trace entirely will result in NaN's; however,
 these zones will not grow in size.

 trin= input trace
 t= time coordinate vector for trin. Trin must be regularly
    sampled. SINCI uses on the first two point in t.
 tout= vector of times for which interpolated amplitudes are
       desired
 trout= output trace. Contains the length(tout) interpolated 
        amplitudes.
 sizetable= size of the sinc function table: [npts,nfuncs]
     where npts = number of points on the sinc function and
     nfuncs = number of uniquely optimized sinc functions. If dt is
	  the input sample interval, then there will be a unique, optimized
	  sinc function designed for interpolation every nfuncs'th of 
	  dt.
   ************* default = [8 25] *********   
   """
   global SINC_TABLE 
   if (sizetable is None):
       sizetable = np.array([8,25])
      
#  see if table needs to be made
       
   try:
       SINC_TABLE
   except NameError:
        maketable = 1
   else:
        maketable = 0
        lsinc,ntable = SINC_TABLE.shape
        
       
   
   """
   lsinc,ntable = SINC_TABLE.shape  # wil throw an error
   if(lsinc*ntable == 0 ):
       maketable = 1
   elif(lsinc !=sizetable[0] or ntable!= sizetable[1]):
       maketable = 1
   """        
   if(maketable ==1 ):
    # Make the sinc function table
       lsinc = sizetable[0]
       ntable = sizetable[1]
    # lsinc should be an even integer
       frac= np.arange(ntable)/ntable
       SINC_TABLE  = np.zeros((lsinc,ntable))
       jmax = np.int(np.fix(ntable/2)+1)
		# the first half of the table is computed by least squares
		# while the second half is derived from the first by symmetry
       for j in range(jmax):
            fmax = np.min(np.array([0.066+ 0.265*np.log(lsinc),1.0]))
            aa = sinque(fmax*np.arange(lsinc))
            tmp = np.arange(lsinc/2-1,-lsinc/2-1,-1)
            bb = fmax*( tmp + frac[j]*np.ones(lsinc))
            cc = sinque(bb)
            matcd = toeplitz(aa.T,aa)
            #SINC_TABLE[:,j] = np.linalg.lstsq(matcd.astype('double'),cc.astype('double'),rcond=None)[0]
            SINC_TABLE[:,j] = (eigenmath.ls_qr(matcd,cc)).flatten()
            #SINC_TABLE[:,j] = np.linalg.solve(matcd,cc)
       point = np.int(lsinc/2)
       jtable = ntable
       ktable = 1
        
       while(SINC_TABLE[point,jtable-1] ==0):
            SINC_TABLE[:,jtable-1] = np.flip(SINC_TABLE[:,ktable],0) # np.fliup
            jtable = jtable-1
            ktable = ktable+1
            
# now interpolate with the tabulated coefficients
# first find the live and dead zone  
   ilive  = mfun.findnot(trin,np.nan)
   if(ilive.size ==0):
       trout = tout*np.nan
       return
   ind  = np.argwhere(np.diff(ilive)>1)
   zone_beg = np.append(ilive[0],ilive[ind+1])
   zone_end = np.append(ilive[ind], ilive[ilive.size-1])
   nzones = zone_beg.size
   dtin = t[1]-t[0]
# now initialize the output trace with nans, then loop over the zones
# and interpolate traces that fall in them
   trout = np.ones(tout.size)*np.nan
   for k in range(nzones):
       # get the input elemets in this zone
       n1 = np.int(np.round((t[zone_beg[k]] -t[0])/dtin)) 
       n2 = np.int(np.round((t[zone_end[k]] -t[0])/dtin) +1)
       ind = np.arange(n1,n2,dtype=int)
       trinzone = trin[ind]
       ii = between(t[zone_beg[k]],t[zone_end[k]],tout,2)
       troutzone = np.ones(ii.size)
       tzone = t[n1:n2]
       if(ii[0]>-1):
           pdata = (tout[ii] - tzone[0])/dtin+1
           deel = pdata - np.fix(pdata)
		# deel now contains the fractional sample increment
		# for each interpolation site
		# compute row number in interpolation table 
           ptable =  np.fix(ntable*deel)
          # compute pointer to inpute data
           pdata = np.fix(pdata) + lsinc/2 - 1
          # pad input data with end values
           tm1 = trinzone[0]* np.ones(np.int(lsinc/2)-1)
           tm2 = trinzone[-1]*np.ones(np.int(lsinc/2))
           trinzone = np.concatenate((tm1,np.concatenate((trinzone,tm2))))
           ij = mfun.find(ptable,ntable+1)
           ptable[ij] = 1*np.ones(ij)+1
           pdata[ij] = pdata[ij]+1
    # finally interpolate by a vector dot product
           for k in range(ii.size):
                tmp = np.arange(pdata[k]-lsinc/2, pdata[k]+lsinc/2, dtype=int)
                troutzone[k] = np.dot(trinzone[tmp] ,SINC_TABLE[:,np.int(ptable[k])])
                if(k==36):
                    pass

           trout[ii] = troutzone
        
   return trout  

#-----------------------------------------------------------------    
def sinci_old(trin,t,tout,sizetable = None):
   """
 SINCI: sinc function interpolation for time series without nan's

  trout=sinci(trin,t,tout,sizetable)

 SINCI performs 8 point sinc function interpolation using a 
 design for the approximate sinc function due to Dave Hale.

 trin= input trace
 t= time coordinate vector for trin. Trin must be regularly
    sampled. SINCI uses only the first two points in t.
 tout= vector of times for which interpolated amplitudes are
       desired
 trout= output trace. Contains the length(tout) interpolated 
        amplitudes.
 sizetable= size of the sinc function table: [npts,nfuncs]
     where npts = number of points on the sinc function and
     nfuncs = number of uniquely optimized sinc functions. If dt is
	  the input sample interval, then there will be a unique, optimized
	  sinc function designed for interpolation every nfuncs'th of 
	  dt.
   ************* default = [8 25] *********  
   """
   global SINC_TABLE 
   if (sizetable is None):
       sizetable = np.array([8,25])
      
#  see if table needs to be made
       
   try:
       SINC_TABLE
   except NameError:
        maketable = 1
   else:
        maketable = 0
        lsinc,ntable = SINC_TABLE.shape
   # initialize trout     
   trout = np.zeros(tout.size,dtype=int)
   
   if(maketable ==1 ):
    # Make the sinc function table
       lsinc = sizetable[0]
       ntable = sizetable[1]
    # lsinc should be an even integer
       frac= np.arange(ntable)/ntable
       SINC_TABLE  = np.zeros((lsinc,ntable))
       jmax = np.int(np.fix(ntable/2)+1)
		# the first half of the table is computed by least squares
		# while the second half is derived from the first by symmetry
       for j in range(jmax):
            fmax = np.min(np.array([0.066+ 0.265*np.log(lsinc),1.0]))
            aa = sinque(fmax*np.arange(lsinc))
            tmp = np.arange(lsinc/2-1,-lsinc/2-1,-1)
            bb = fmax*( tmp + frac[j]*np.ones(lsinc))
            cc = sinque(bb)
            matcd = toeplitz(aa.T,aa)
            #SINC_TABLE[:,j] = np.linalg.lstsq(matcd.astype('double'),cc.astype('double'),rcond=None)[0]
            SINC_TABLE[:,j] = (eigenmath.ls_qr(matcd,cc)).flatten()
            #SINC_TABLE[:,j] = np.linalg.solve(matcd,cc)
       point = np.int(lsinc/2)
       jtable = ntable
       ktable = 1
        
       while(SINC_TABLE[point,jtable-1] ==0):
            SINC_TABLE[:,jtable-1] = np.flip(SINC_TABLE[:,ktable],0) # np.fliup
            jtable = jtable-1
            ktable = ktable+1   
 
# now interpolate with the tabulated coefficients
# first extrapolate with constant end values
# for beginning:
   ii = mfun.le(tout,t[0])  
   if(ii.size !=0):
       trout[ii] = trin[0]*np.ones(ii.size)
   ii = mfun.ge(tout,t[-1])       
   if(ii.size !=0):
       trout[ii] = trin[0]*np.ones(ii.size) 
   # intermediate samples
   dtin = t[1]-t[0]
   ii = np.intersect1d(mfun.gt(tout,t[0]), mfun.lt(tout,t[-1]))
   if(ii.size !=0):
       pdata = (tout[ii]-t[0])/dtin+1
       deel = pdata - np.fix(pdata)
	# del now contains the fractional sample increment
	# for each interpolation site
	# compute row number in interpolation table       
       ptable = mfun.mround(ntable*deel) 
       #compute pointer to input  data
       pdata = np.fix(pdata) + lsinc/2 - 1
       # pad input data with end values
       tm1 = trin[0]* np.ones(np.int(lsinc/2)-1)
       tm2 = trin[-1]*np.ones(np.int(lsinc/2))
       trin = np.concatenate((tm1,np.concatenate((trin,tm2))))
       ij = mfun.find(ptable,ntable+1)
       ptable[ij] = 1*np.ones(ij.size)
       pdata[ij] = pdata[ij]+1
    # finally interpolate by a vector dot product
       for k in range(ii.size):
           tmp = np.arange(pdata[k]-lsinc/2, pdata[k]+lsinc/2, dtype=int)
           trout[ii[k]] = np.dot(trin[tmp] ,SINC_TABLE[:,np.int(ptable[k])])       
       
       return trout      
    
#-----------------------------------------------------------------
def sinque(x):
    eps =2.2204e-16
    s = np.zeros(x.size)
    ii = np.argwhere(np.abs(x) <=eps)  
    if (ii.size !=0):
        s[ii] = np.ones(ii.size)
        ii = mfun.findnot(s,1.0)
        s[ii] = np.sin(x[ii])/x[ii]
    else:
        s = np.sin(x)/x

    return s   
#***********************************************************
if __name__ == "__main__":  
    import matplotlib.pyplot as plt
    sig = 1000*np.array([1,4,7,5,7,1,10,5,7,0.1,0,1,3,5,1])
    t = 2* np.arange(sig.size)
    sig2,t2 = resamp_crewes(sig,t,0.5)
    #plt.plot(sig,t)
    #plt.plot(sig2,t2)
    plt.close('all')
    xx = sinci_old(sig,t,t2)
    plt.plot(sig,t)
    plt.plot(xx,t2)
    
   
   
            
            
            
    
    
    
    