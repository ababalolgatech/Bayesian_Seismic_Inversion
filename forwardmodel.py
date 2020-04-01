# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:27:53 2019

@author: Ayodeji Babalola
"""
import numpy as np
import math
import matlab_funcs as mfun
import Filter as ft
"""
def convmtx(kern,nsamp):
    ncol = len(kern)
    nrow = ncol + nsamp-1
    
    convmat = np.zeros((nrow,nrow))
    k=0
    
    for i in range (0,nrow):
        ind = ncol + k         
        if ind > nrow:
            ind = nrow 
            
            for ii in range (0,ind):
                convmat[ii][i] = kern[ii-k]
        k = k+1
    return convmat

"""
#------------------------------------------------------------------------------
def convmtx(v,n):
    """Generates a convolution matrix
    
    Usage: X = convm(v,n)
    Given a vector v of length N, an N+n-1 by n convolution matrix is
    generated of the following form:
              |  v(0)  0      0     ...      0    |
              |  v(1) v(0)    0     ...      0    |
              |  v(2) v(1)   v(0)   ...      0    |
         X =  |   .    .      .              .    |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |  v(N) v(N-1) v(N-2) ...  v(N-n+1) |
              |   0   v(N)   v(N-1) ...  v(N-n+2) |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |   0    0      0     ...    v(N)   |
    And then it's trasposed to fit the MATLAB return value.     
    That is, v is assumed to be causal, and zero-valued after N.
    """
    N = len(v) + 2*n - 2
    xpad = np.concatenate([np.zeros(n-1), v[:], np.zeros(n-1)])
    X = np.zeros((len(v)+n-1, n))
    # Construct X column by column
    for i in range(n):
        X[:,i] = xpad[n-i-1:N-i]
    
    return X.transpose()
#------------------------------------------------------------------------------
def convmat2(sig,nsamp):
    wav1 = convmtx(sig,nsamp)
    convmat = wav1[0:nsamp,0:nsamp]
    return convmat
#------------------------------------------------------------------------------
def convmat1(sig,nsamp):
    nn2 = np.int(sig.size/2)
    wav1 = convmtx(sig,nsamp)
    convmat = wav1[nn2:nn2-1+nsamp,0:nsamp]
    return convmat
#------------------------------------------------------------------------------
def Der(nn) :
    
    Der = np.zeros((nn,nn))
    for i in range(0,nn-1):
        Der[i][i] = -1   
        Der[i][i+1]= 1   
     
        Der[nn-1][nn-1] = -1 ;
    return Der
#------------------------------------------------------------------------------
def avopp(vp1,vs1,rho1,vp2,vs2,rho2,ang,algo):

    t = np.divide(ang*np.pi,180)
    pf = np.sin(t)/vp1      
    ct = np.cos(t)     
    da = 0.5*(rho1+rho2)
    Dd = rho2-rho1
    vpa = 0.5*(vp1+vp2)
    Dvp = vp2-vp1
    vsa = 0.5*(vs1+vs2)
    Dvs = vs2-vs1
    
    
    if algo == 1: # FULL Zoepfritz (A&K)
        ct2 = np.sqrt(1-(np.sin(t)**2*(vp2**2/vp1**2)))             
        cj1 = np.sqrt(1-(np.sin(t)**2*(vs1**2/vp1**2)))             
        cj2 = np.sqrt(1-(np.sin(t)**2*(vs2**2/vp1**2)))             
        a   = (rho2*(1-(2*vs2**2*pf**2)))-(rho1*(1-(2*vs1**2*pf**2))) 
        b   = (rho2*(1-(2*vs2**2*pf**2)))+(2*rho1*vs1**2*pf**2)       
        c   = (rho1*(1-(2*vs1**2*pf**2)))+(2*rho2*vs2**2*pf**2)       
        d   = 2*((rho2*vs2**2)-(rho1*vs1**2))                          
        E   = (b*ct/vp1)+(c*ct2/vp2)                                  
        F   = (b*cj1/vs1)+(c*cj2/vs2)                                 
        G   = a-(d*ct*cj2/(vp1*vs2))                                  
        H   = a-(d*ct2*cj1/(vp2*vs1))                                 
        D   = (E*F)+(G*H*pf**2)                                         
        Rpp = ( (((b*ct/vp1)-(c*ct2/vp2))*F) - 
        ((a+(d*ct*cj2/(vp1*vs2)))*H*pf**2) ) / D                              
    elif algo ==2: #  AKI & RICHARDS
        Rpp =(0.5*(1-(4*pf**2*vsa**2))*Dd/da) + (Dvp/(2*ct**2*vpa))- (4*pf**2*vsa*Dvs)  
    elif algo ==3: # SHUEY
        poi1 = ((0.5*(vp1/vs1)**2)-1)/((vp1/vs1)**2-1)                 
        poi2 = ((0.5*(vp2/vs2)**2)-1)/((vp2/vs2)**2-1)                 
        poia = (poi1+poi2)/2
        Dpoi=(poi2-poi1)                          
        Ro   = 0.5*((Dvp/vpa)+(Dd/da))                                  
        Bx   = (Dvp/vpa)/((Dvp/vpa)+(Dd/da))                           
        Ax   = Bx-(2*(1+Bx)*(1-2*poia)/(1-poia))                       
        Rpp  = Ro + (((Ax*Ro)+(Dpoi/(1-poia)**2))*np.sin(t)**2) + (0.5*Dvp*(np.tan(t)**2-np.sin(t)**2)/vpa)                    
        
    elif algo ==4:     # SHUEY LINEAR (2 TERM)
        A  = 0.5*((Dvp/vpa)+(Dd/da))                                    
        B  = (-2*vsa**2*Dd/(vpa**2*da)) + (0.5*Dvp/vpa) - (4*vsa*Dvs/(vpa**2))                                          
        Rpp = A+(B*np.sin(t)**2)                                             
            
    return Rpp

    
#------------------------------------------------------------------------------
def datnormhrs(Rdepth,seis,tseis,Vp,Vs,Rho,tlog,wav,AVO):
    num = 10
    if (seis.shape[1] != wav.size):
        raise Exception ("seismic and obj.wavletelet has unequal angles")
    dat  = mfun.segdat(Rdepth,tseis,seis)   # to do    
    vp,vs,rho = mfun.seglog(Rdepth,tlog,Vp,Vs,Rho) # to do 
    ndat,nang = mfun.size(wav)
    sc = mfun.cell(nang)
    datnorm = mfun.cell(ndat,nang)
    Re = Dobsn(vp,vs,rho,AVO,ang)  # to do 
    
    sc = np.zeros(nang)
    for i in range(nang):
        synth = convm(Re[i],wav[i])
        Logsc = np.sort(ft.xcorr(wav[i],synth))  # BUG
        Trcsc = np.sort(ft.xcorr(wav[i],dat[i]))  # BUG
        
        wscl,lag = ft.xcorr(wav[i])
        wsc = wscl[np.nonzero(lag ==0)]
        Logsc = Logsc/wsc
        nsL = len(Logsc)
        nsT = len(Trcsc)
        dLg = mfun.rms(np.concatenate(Logsc[0:num],Logsc[num:-1]))
        dTr = mfun.rms(np.concatenate(Trcsc[0:num],Trcsc[num:-1]))
        
        sc[i] = dTr/dLg
        datnorm = seis[i]/sc[i]
        
    return sc,datnorm
        
#------------------------------------------------------------------------------
def dobsn(vp,vs,rho,ang,AVO):
    nr,nc = mfun.shape(vp)
    nang = len(ang)
    
    R = np.zeros(nr,1)
    Re = mfun.cell(nang)
    
    for ii in range(nang):
        for i in range(nr-1):
            Rpp = avopp(vp[i],vs[i],rho[i], \
                        vp[1+i],vs[1+i],rho[1+i], \
                        ang[ii],AVO)
            R[i] = np.real(Rpp)
        
            R[nr-1] = R[nr-2]
        Re[ii] = R
        
    return Re
    
#------------------------------------------------------------------------------
def convm(Re,wavelet):  # should be in filter module
    sig = np.convolve(Re,wavelet[0,i],'same')
      #  signal.convolve(Re[i],self.wavelet[i],mode='same')
    return sig
        
        


            
        
        
            
            
            
            
            
            
            
            
            
        
        

#------------------------------------------------------------------------------
if __name__ == "__main__":
    # testing 
    cov = Der(3)
    vp1 = 5000
    vp2 = 2000
    vs1 = 3500
    vs2 = 1500
    rho1 = 2.8
    rho2 = 2.6
    ang = np.array([0,45,90])
    R = avopp(vp1,vs1,rho1,vp2,vs2,rho2,ang,4)    
    h = [1,2,3,4,5]
    Xd = convmat2(h,7)
    print(Xd)
    """
    sig = np.random.uniform(-3,3,5)
    sig = np.array([1,2,3,4,5])
    xx = convmtx(sig,5)
    
   

    if __name__ == '__main__':
        main()
    """