# -*- coding: utf-8 -*-
"""
Basic rock-physics routines

@author: Dr. Ayodeji Babalola
"""
#from numba import jit
from scipy import signal
import numpy as np
from scipy import interpolate
from warnings import warn
#from scipy.linalg.fblas import dgemm
from scipy.linalg.blas import dgemm
 #----------------------------------------------------------------------------
def resample(data,time,dtold,dtnew,options = None):
    if (dtold == dtnew):
        tcord = time
        dat = data
    else:
        sc = dtold/dtnew
        ns = round(sc*data.size)
        tmin = np.min(time)
        tmax = np.max(time)
        tcord = np.linspace(tmin,tmax,ns,endpoint=False)
        """
        
        datt = mfun.padd(data,pdns*dtnew)
        y =signal.resample(datt,ns)
        dat = mfun.upadd(y,pdns)
        """
        if (options is None):
            dat = resamp_by_interp(data,dtold,dtnew)
            ns = dat.shape[0]
            tmin = np.min(time)
            tmax = np.max(time)
            tcord = np.linspace(tmin,tmax,ns,endpoint=False)
            
        elif(options == "fft" ):
            dat = signal.resample(data,ns)
        elif(options == "polyphase"):
             dat = signal.resample_poly(data,ns)
        
    return dat,tcord
 #----------------------------------------------------------------------------
def resamp_by_interp(signal,dtold, dtnew):

    scale =  dtold/dtnew
    # calculate new length of sample
    n = round(len(signal) * scale)

    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),            # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal
#----------------------------------------------------------------------------
def resamp_by_newlength(sig,new_length):
    ns_in = sig.size
    ns_out = ns_in/new_length
    data = resamp_by_interp(sig,1,ns_out)
    return data
    
 #----------------------------------------------------------------------------
#@jit(nopython=True)
def Filt(data,freq,dt = None):
    
    """
Filtering function
[dat] = Filt(data, freq,dt)
dt must be in milliseconds
wind is filter length

    """
    if (dt is None):
        dt = 1
    period = round(1000/freq)
    wind = int(round(period/dt))
    dat = movingmean(data,wind)
    return dat
 #---------------------------------------------------------------------------
#@jit(nopython=True)  
def movingmean(data,window):
  
    """
    Calculates the centered moving average of an n-dimensional matrix in any direction. 
       result=movingmean(data,window,dim,option)
    
       Inputs: 
       1)  data = The matrix to be averaged. 
       2)  window = The window size.  This works best as an odd number.  If an even 
           number is entered then it is rounded down to the next odd number.  
       3)  dim = The dimension in which you would like do the moving average.
           This is an optional input.  To leave blank use [] place holder in
           function call.  Defaults to 1.
       4)  option = which solution algorithm to use.  The default option works
           best in most situations, but option 2 works better for wide
           matrices (i.e. 1000000 x 1 or 10 x 1000 x 1000) when solved in the
           shorter dimension.  Data size where option 2 is more efficient will 
           vary from computer to computer. This is an optional input.  To leave  
           blank use [] place holder in function call.  Defaults to 1.
     
       Example:  
       Calculate column moving average of 10000 x 10 matrix with a window size 
       of 5 in the 1st dimension using algorithm option 1.
       d=rand(10000,10);
       dd=movingmean(d,5,1,1);
               or
       dd=movingmean(d,5,[],1);
               or
       dd=movingmean(d,5,1,[]);
               or
       dd=movingmean(d,5,1);
               or
       dd=movingmean(d,5,[],[]);
               or
       dd=movingmean(d,5);
    
        """
    
    
    dim = 1        
    option = 1 # add options later when you can code with varargin
    
    if window % 2 == 0 :
        window = window-1
    
    """
    Calculates the number of elements in before and after the central element
    %to incorporate in the moving mean.  Round command is just present to deal
    %with the potential problem of division leaving a very small decimal, ie.
    %2.000000000001.
    """
    halfspace=round((window-1)/2);
    
    # %calculates the size of the input data set
    n = len(data)
    
    if window % 2 ==0 :
       window=window-1
    
    if n<window :
        raise Exception('window is too large')
    
    
    
    #if ndims(data)<=2 :
        """
        Solution for 1d-2d situation.  Uses vector operations to optimize
        solution speed.
        
        To simplify algorithm the problem is always solved with dim=1.
        If user input is dim=2 then the data is transposed to calculate the
        solution in dim=1
    
        if dim==2 :
            data=data';
            """
            
    """
        The three best solutions I came up to for the 1d-2d problem.  I kept
        them in here to preserve the code incase I want to use some of it
        again.  I have found that option 1 is consistenetly the fastest.
        option=1;
    """

    if option == 1:
        """
        option 1, works best for most data sets
                
                Computes the beginning and ending column for each moving
                average compuation.  Divide is the number of elements that are
                incorporated in each moving average.            
        """
         
        start = np.concatenate((np.ones(halfspace+1),
        np.linspace(2,n-halfspace,(n-halfspace-1)))) 
                
        stop=np.concatenate((np.linspace((1+halfspace),n,(n-halfspace)),
        np.ones(halfspace)*n))             
                       
        stop = stop.astype(int) -1   # you can pre-alllocate as int to remove confusion with the  int
        start = start.astype(int)-1
        divide=stop-start+1;
                
        """
                Calculates the moving average by calculating the sum of elements
                from the start row to the stop row for each central element,
                and then dividing by the number of elements used in that sum
                to get the average for that central element.
                Implemented by calculating the moving sum of the full data
                set.  Cumulative sum for each central element is calculated by
                subtracting the cumulative sum for the row before the start row
                from the cumulative sum for the stop row.  Row references are
                adusted to take into account the fact that you can now
                reference a row<1.  Divides the series of cumulative sums for
                by the number of elements in each sum to get the moving
                average.        
        """
        CumulativeSum=np.cumsum(data)        
        ind = matlab_max(start-1,0)
        #ind = ind.astype(int)                
        temp_sum=CumulativeSum[stop]- CumulativeSum[ind]
        # the expression in matlabs is max(start-1,1) -- figure it out later 
                #  temp_sum((start==1),:)=bsxfun(@plus,temp_sum((start==1),:),data(1,:));
        temp_sum[start==0] = temp_sum[(start==0)] + data[0]
               # result=bsxfun(@rdivide,temp_sum,divide');   
        result = np.divide(temp_sum,divide)
                
    elif option == 2 :
        """
                option 2, Can be faster than option 1 in very wide matrices
                (i.e. 100 x 1000 or l000 x 10000) when solving in the shorter
                dimension, but in general it is slower than option 1.
                Uses a for loop to calculate the data from the start row to
                the stop row, and then divides by the number of rows
                incorporated.
               
                result=zeros(size(data));
                for i=1:n(dim)
                    start=max(1,i-halfspace);
                    stop=min(i+halfspace,n(dim));
                    result(i,:)=sum(data(start:stop,:),1)/(stop-start+1);
                end    
        """ 

    return result
#------------------------------------------------------------------------------
def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")
    
    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html        
    """
    
    if len(x) != len(s):
        raise Exception( 'x and s must be the same length')
    
    # Find the period    
    T = s[1] - s[0]
    
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sincM/T))
    return y
#------------------------------------------------------------------------------
def interp3 (x,xt,xp,workers = None):
  """
  Interpolate the signal to the new points using a sinc kernel

  Like interp, but splits the signal into domains and calculates them
  separately using multiple threads.

  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on
  workers  number of threaded workers to use (default: 16)

  output:
  y     the interpolated signal at points xp
  """

  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = len(xp)

  y = np.zeros((m, nn))

  # from upsample
  if workers is None: workers = 6

  xxp = np.array_split (xp, workers)

  from concurrent.futures import ThreadPoolExecutor
  import concurrent.futures

  def approx (_xp, strt):
    for (pi, p) in enumerate (_xp):
      si = np.tile (np.sinc (xt - p), (m, 1))
      y[:, strt + pi] = np.sum (si * x)

  jobs = []
  with ThreadPoolExecutor (max_workers = workers) as executor:
    strt = 0
    for w in np.arange (0, workers):
      f = executor.submit (approx, xxp[w], strt)
      strt = strt + len (xxp[w])
      jobs.append (f)


  concurrent.futures.wait (jobs)

  return y.squeeze ()

#------------------------------------------------------------------------------
def scipy_interp(dat,x,xnew):
    f = interpolate.interp1d(dat,x,kind = 'linear',fill_value="extrapolate")
    dat_new = f(xnew)
    return dat_new
#------------------------------------------------------------------------------
def matlab_max (var1,var2):
    nvar1 = len(var1)
#  nvar2 = len(var2)
    dat = np.zeros(nvar1).astype(int) 
#    if nvar1 ==1 : # if var2 is a scalar
    for i in range(0,nvar1):
          if var1[i]> var2:
             dat[i] = var1[i]
          else:
             dat[i] = var2
#    else:
#        for i in range(0,nvar1):
#            if var1[i]> var2:
#                dat[i] = var1[i]
#            else:
#                dat[i] = var2[i]
                
    """
     what  if the variables are reverserd ?
    """           
    return dat
#------------------------------------------------------------------------------
def backusaveraging(vp,vs,rho,fdom,dtlog):
   #print('needs testing')
   """
[vpb,vsb,rhob] = backusaveraging(vp,vs,rho,fdom,dtlog)
 Please use highcut frequency
I am trying to pick thin layers within seismic frequncy
 Algorithm will not work if you use fdom
   """
   Vp = vp
   Vs = vs
   den = rho
   Vs2 = Vs**2
   Vp2 = Vp**2
   nnd = den*Vp2
   mmm = den*Vs2
   
   
   # calculating running averages
   o_den =  Filt(den,fdom,dtlog)
   o_mmm =  Filt(mmm,fdom,dtlog)   
   o_nnd =  Filt(nnd,fdom,dtlog)     
   
   
   #  CALCULATING EFFECTIVE PARAMETERS
   vsb = np.sqrt(o_mmm/o_den)
   vpb = np.sqrt(o_nnd/o_den)
   rhob = o_den

   return vpb,vsb,rhob

#-----------------------------------------------------------------------------
def convm(sig1,sig2,pct = None):
    if (pct is None):
        pct = 10
    #covm = signal.convolve(sig1,sig2,mode='same')
    dat = signal.convolve(sig1,sig2)
    ns = sig1.size
    if (pct > 0):
        mw = mwhalf(ns,pct)
    else:
        mw = np.ones(ns)
   
    dat = dat[0:ns]*mw
    return dat

#-----------------------------------------------------------------------------
def xcorr(sig1,sig2= None):
    if (sig2 is None):
        sig2 = sig1 # autocorrelation
        
    corr = np.correlate(sig1,sig2,mode ='full')
    maxlag = np.argmax(corr)    
#    covm = signal.convolve(sig1,sig2,mode='same')
    return corr,maxlag

#-----------------------------------------------------------------------------
def xcorr_matlab(sig1,sig2= None):
    if (sig2 is None):  # mimicks matlab xcorr
        sig2 = sig1 # autocorrelation
    
    n1 = sig1.size
    n2 = sig2.size

    if (n1!=n2):
        if(n1 > n2):
            tmz = np.zeros(n1-n2)
            sig2 = np.append(sig2,tmz)
        elif(n2>n1):
            tmz = np.zeros(n2-n1)
            sig1 = np.append(sig1,tmz)            
        
    corr = np.correlate(sig1,sig2,mode ='full')
    maxlag = np.argmax(corr)    
#    covm = signal.convolve(sig1,sig2,mode='same')
    return corr,maxlag

#------------------------------------------------------------------------------   
def rms(sig):
    return np.sqrt(np.mean(np.square(sig)))

#------------------------------------------------------------------------------ 
#@jit(nopython=True)    
def mwhalf(n,percent=None):
    # half an mwindow (boxcar with raised-cosine taper on one end)
    if (percent is None):
        percent =10   
    if(percent>100 or percent<0 ):
        raise Exception ('invalid percent for mwhalf')
    
    m = np.int(np.floor(percent*n/100))
    h = signal.windows.hann(2*m)
    tmp = np.ones(n-m)
    indx = np.arange(m,0,-1)
    aa = h[indx]
    dat = np.hstack((tmp,aa)) 
    return dat
    
#------------------------------------------------------------------------------   
def xcorr2(sig1,sig2):
    tmp = signal.correlate(sig1,sig2)
    a,b = np.diag(tmp)
    return a,b

#------------------------------------------------------------------------------
def lsq(a,b,residuals=False):   
    if type(a) != np.ndarray or not a.flags['C_CONTIGUOUS']:
        warn('Matrix a is not a C-contiguous numpy array. The solver will create a copy, which will result' + \
             ' in increased memory usage.')
    a = np.asarray(a, order='c')
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i,dgemm(alpha=1.0, a=a.T, b=b)).flatten()

    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x
#----------------------------------------------- 
if __name__ == "__main__": 
    w = mwhalf(200,10)
    import matplotlib.pyplot as plt
  
    xx = np.array([2, 3, 4,100])
    dat =resamp_by_newlength(xx,5)
    

    t = np.linspace(0, 10, 50, endpoint=False)
    y = np.cos(-t**2/6.0)  
    
    #y1,t1 = resample(y,t,1,2,"fft")
    y2,t2 = resample(y,t,0.02,0.001) # best
    plt.plot(y2,t2)
    print( y.size)
    print(y2.size)
    """
    
    sig1 = np.array([1,2,3])
    sig2 = np.array([10,0.2,7])
    
    xcorr(sig1,sig2)
      """




