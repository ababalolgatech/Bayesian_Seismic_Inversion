# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:55:31 2019

@author: Dr. Ayodeji Babalola
"""
from scipy.optimize import minimize
from scipy.optimize import leastsq
from lmfit import minimize, Parameters
from lmfit import Minimizer, Parameters, report_fit
import numpy as np
import matlab_funcs as mfun
import Filter as ft


#----------------------------     
def compactntrend_AI(z,data,params):
        vpyoung  = 3000        
        #mod =   vpyoung  + params['a']*np.exp(params['b']*z )
        mod =   vpyoung  + (z**3 * params['a']) + (params['b']* z**2) + (params['c']* z) 
        residual = np.absolute(data - mod)**2
        return residual
def fcompactntrend_AI(z,params):
        vpyoung  = 3000        
        #mod =   vpyoung  + params['a']*np.exp(params['b']*z )
        mod =   vpyoung  + (z**3 * params['a']) + (params['b']* z**2) + (params['c']* z) 
 
        return mod    
    
def compactntrend_rho(z,data,params):
        water = 2.2
        #mod =   water + params['a']*np.exp( params['b']*z)      
        mod =   water +(z**2 * params['a']) + (params['b']* z)    
        residual = (data - mod)**2
        return residual
        
def fcompactntrend_rho(z,params):
        water = 2.2
        #mod =   water + params['a']*np.exp( params['b']*z)      
        mod =   water +(z**2 * params['a']) + (params['b']* z)        
        return mod         
        
#----------------------------     
def compactntrend_vp(z,data,params):
        vpyoung  = 1500        
        #mod =   vpyoung  + params['a']*np.exp(params['b']*z )
        mod =   vpyoung  + (z**3 * params['a']) + (params['b']* z**2) + (params['c']* z) 
        residual = np.absolute(data - mod)**2
        return residual
        
def fcompactntrend_vp(z,params):
        vpyoung  = 1500        
        #mod =   vpyoung  + params['a']*np.exp(params['b']*z )
        mod =   vpyoung  + (z**3 * params['a']) + (params['b']* z**2) + (params['c']* z) 
 
        return mod        
        
#----------------------------     
def compactntrend_vs(z,data,params):
        vsyoung  = 1500        
        #mod =   vsyoung + params['a']*np.exp(params['b']* z) + params['c']*np.exp( params['d']*z )  
        mod =   vsyoung + (z**3 * params['a']) + (params['b']* z**2) + (params['c']* z)  
        residual = (data - mod)**2
        return residual
        
def fcompactntrend_vs(z,params):
        vsyoung  = 1500        
        #mod =   vsyoung + params['a']*np.exp(params['b']* z) + params['c']*np.exp( params['d']*z )  
        mod =   vsyoung + (z**3 * params['a']) + (params['b']* z**2) + (params['c']* z)           
        return mod

#----------------------------     
def compactntrend_gr(z,data,params):
        vsyoung  = 50        
        #mod =   vsyoung + params['a']*np.exp(params['b']* z) + params['c']*np.exp( params['d']*z )  
        mod =   vsyoung + (z**2 * params['a']) + (params['b']* z)   
        residual = (data - mod)**2
        return residual
        
def fcompactntrend_gr(z,params):
        vsyoung  = 50        
        #mod =   vsyoung + params['a']*np.exp(params['b']* z) + params['c']*np.exp( params['d']*z )  
        mod =   vsyoung + (z**2 * params['a']) + (params['b']* z)        
        return mod        
#----------------------------    
      
def residual(params,data,z,wlog):      
        if (wlog == 'AI'):
            residual = compactntrend_AI(z,data,params)      
        elif (wlog == 'Vp'):
            residual = compactntrend_vp(z,data,params)            
        elif(wlog == 'Vs'):
            residual = compactntrend_vs(z,data,params)            
        elif(wlog == 'Rhob'):
            residual = compactntrend_rho(z,data,params)
        elif(wlog == 'GR'):
            residual = compactntrend_gr(z,data,params)            
        else:
            raise Exception ('not yet coded for this lognames')
        return residual
#----------------------------    
def ctrendlogs(data,z,wlog):
        print('BUG: Vs predicton is wrong')
        print('--------------------------')
        indnan = np.argwhere(np.isnan(data))
        if (indnan.size >0  ):
            msg = wlog + ' stil has nan values but is removed for now'
            print('BUG: inside ctrendlog module')
            print(msg)
            print('--------------------------------------------------')
            data = np.delete(data,indnan)
            z = np.delete(z,indnan)
            dtold =  z[1]- z[0]
            data = ft.Filt(data,10,dtold)

        
        params = Parameters()
        lskws = dict(ftol=1.e-20, xtol=1.e-20) 
        # create Minimizer
        if (wlog == 'AI'):  # not tested
            scal = 1   
            data = data/scal            
            params.add('a',value = 1.0000e-10,min =0,max = 300,vary = True)
            params.add('b',value = 1.0000e-10,min =0,max = 300,vary = True)   
            params.add('c',value = 1.0000e-10,min =0,max = 300,vary = True)
            mini = Minimizer(residual, params,fcn_args =(data,z,wlog))
            result = mini.minimize(method='least_squares', **lskws)        
        elif (wlog == 'Vp'):
            scal = 1   
            data = data/scal            
            params.add('a',value = 1.0000e-10,min =0,max = 300,vary = True)
            params.add('b',value = 1.0000e-10,min =0,max = 300,vary = True)   
            params.add('c',value = 1.0000e-10,min =0,max = 300,vary = True)
            mini = Minimizer(residual, params,fcn_args =(data,z,wlog))
            result = mini.minimize(method='least_squares', **lskws)
        elif (wlog == 'Vs'):
            scal = 1 ;            
            data = data/scal
            params.add('a',value = 1.0000e-10,min =0,max = 0.005,vary = True)
            params.add('b',value = 1.0000e-10,min =0,max = 0.001,vary = True)
            params.add('c',value = 1.0000e-10,min =0,max = 0.900,vary = True)
            params.add('d',value = 1.0000e-10,min =0,max = 0.002,vary = True)            
            mini = Minimizer(residual, params,fcn_args=(data,z,wlog))
            result = mini.minimize(method='least_squares', **lskws)   
        elif (wlog == 'GR'):
            scal = 1 ;            
            data = data/scal
            params.add('a',value = 1.0000e-10,min =0,max = 0.005,vary = True)
            params.add('b',value = 1.0000e-10,min =0,max = 0.001,vary = True)         
            mini = Minimizer(residual, params,fcn_args=(data,z,wlog))
            result = mini.minimize(method='least_squares', **lskws)             
        elif (wlog == 'Rhob'):
            params.add('a',value = 1.0000e-10,min =0,max = 1.0000,vary = True)
            params.add('b',value = 1.0000e-10,min =0,max = 0.00012,vary = True)             
            mini = Minimizer(residual, params,fcn_args=(data,z,wlog))           
            result = mini.minimize(method='least_squares', **lskws) 

        return result.params
#---------------------------- 
def predict(params,wlog,LOGS,tseis) :
        dt  = tseis[1] - tseis[0]
        nwells,nlogs = LOGS.size    
        logs  = mfun.cell(nwells,nlogs)
        for i in range(nwells):
            t1 = np.min(tseis)
            t2 = np.min(LOGS[i,0]) -dt
            upper = np.arange(t1,t2+dt,dt)
            z = upper
            if (upper.size == 0):
                upperzone_data = z  # there is data above the horizon                
            else:
                if (wlog == 'AI'): 
                    scal = 1
                    upperzone_data =  scal*fcompactntrend_AI(z,params)                
                elif (wlog == 'Vp'): 
                    scal = 1
                    upperzone_data =  scal*fcompactntrend_vp(z,params)
                elif (wlog == 'Vs'): 
                    scal = 1
                    upperzone_data =  scal*fcompactntrend_vs(z,params)            
                elif (wlog == 'Rhob'):                
                    upperzone_data = fcompactntrend_rho(z,params) 
                elif (wlog == 'GR'):                
                    upperzone_data = fcompactntrend_gr(z,params)                 
                else:
                    raise Exception ('not yet coded for this lognames')
                    
            t2 = np.max(tseis)
            t1 = np.max(LOGS[i,0])+dt
            lower = np.arange(t1,t2+dt,dt)
            z = lower
            if (lower.size ==0):
                lowerzone_data = lower  # there is data beneath tseis
            else:
                if (wlog == 'AI'):
                    scal = 1
                    lowerzone_data =  scal*fcompactntrend_AI(z,params)                
                elif (wlog == 'Vp'):  
                    scal = 1
                    lowerzone_data =  scal*fcompactntrend_vp(z,params)
                elif (wlog == 'Vs'):
                    scal=1
                    lowerzone_data =  scal*fcompactntrend_vs(z,params)            
                elif (wlog == 'Rhob'):  
                    scal = 1
                    lowerzone_data =  fcompactntrend_rho(z,params) 
                elif (wlog == 'GR'):                
                    lowerzone_data =  fcompactntrend_gr(z,params)                 
                else:
                    raise Exception ('not yet coded for this lognames') 
                
            if (upper.size == 0 and lower.size !=0):
                zone = np.array([np.min(tseis),LOGS[i,0][-1]],dtype=int)
                data,t = mfun.segment_logs(zone,LOGS[i,0],LOGS[i,1])
                dat_out = np.concatenate((data,lowerzone_data),axis = None)
            elif(upper.size != 0 and lower.size ==0):
                zone = np.array([LOGS[i,0][0],tseis[-1]],dtype=int) 
                data,t = mfun.segment_logs(zone,LOGS[i,0],LOGS[i,1])
                dat_out = np.concatenate((upperzone_data,data),axis = None) 
            elif(upper.size == 0 and lower.size ==0):
                zone = np.array([np.min(tseis),np.max(tseis)],dtype=int) 
                data,t = mfun.segment_logs(zone,LOGS[i,0],LOGS[i,1])
                dat_out = data                
            else:  #upper and lower zone zero               
                dat_out = np.concatenate((upperzone_data,LOGS[i,1],lowerzone_data),axis = None)
            
            logs[i,0] = tseis
            logs[i,1] = dat_out
            
        return logs
#---------------------------- 
def predict_ver2(params,wlog,LOGS,tseis) :
        dt  = tseis[1] - tseis[0]
        nwells,nlogs = LOGS.size    
        logs  = mfun.cell(nwells,nlogs)
        for i in range(nwells):
            t1 = np.min(tseis)
            t2 = np.min(LOGS[i,0]) -dt
            upper = np.arange(t1,t2+dt,dt)
            z = upper
            if (upper.size == 0):
                upperzone_data = z  # there is data above the horizon                
            else:
                if (wlog == 'AI'): 
                    scal = 1
                    upperzone_data =  scal*fcompactntrend_AI(z,params)                
                elif (wlog == 'Vp'): 
                    scal = 1
                    upperzone_data =  scal*fcompactntrend_vp(z,params)
                elif (wlog == 'Vs'): 
                    scal = 1
                    upperzone_data =  scal*fcompactntrend_vs(z,params)            
                elif (wlog == 'Rhob'):                
                    upperzone_data = fcompactntrend_rho(z,params) 
                elif (wlog == 'GR'):                
                    upperzone_data = fcompactntrend_gr(z,params)                 
                else:
                    raise Exception ('not yet coded for this lognames')
                    
            t2 = np.max(tseis)
            t1 = np.max(LOGS[i,0])+dt
            lower = np.arange(t1,t2+dt,dt)
            z = lower
            if (lower.size ==0):
                lowerzone_data = lower  # there is data beneath tseis
            else:
                if (wlog == 'AI'):
                    scal = 1
                    lowerzone_data =  scal*fcompactntrend_AI(z,params)                
                elif (wlog == 'Vp'):  
                    scal = 1
                    lowerzone_data =  scal*fcompactntrend_vp(z,params)
                elif (wlog == 'Vs'):
                    scal=1
                    lowerzone_data =  scal*fcompactntrend_vs(z,params)            
                elif (wlog == 'Rhob'):  
                    scal = 1
                    lowerzone_data =  fcompactntrend_rho(z,params) 
                elif (wlog == 'GR'):                
                    lowerzone_data =  fcompactntrend_gr(z,params)                 
                else:
                    raise Exception ('not yet coded for this lognames') 
                
            if (upper.size == 0 and lower.size !=0):
                zone = np.array([np.min(tseis),LOGS[i,0][-1]],dtype=int)
                data,t = mfun.segment_logs(zone,LOGS[i,0],LOGS[i,3])
                dat_out = np.concatenate((data,lowerzone_data),axis = None)
            elif(upper.size != 0 and lower.size ==0):
                zone = np.array([LOGS[i,0][0],tseis[-1]],dtype=int) 
                data,t = mfun.segment_logs(zone,LOGS[i,0],LOGS[i,3])
                dat_out = np.concatenate((upperzone_data,data),axis = None) 
            elif(upper.size == 0 and lower.size ==0):
                zone = np.array([np.min(tseis),np.max(tseis)],dtype=int) 
                data,t = mfun.segment_logs(zone,LOGS[i,0],LOGS[i,3])
                dat_out = data                
            else:  #upper and lower zone zero               
                dat_out = np.concatenate((upperzone_data,LOGS[i,3],lowerzone_data),axis = None)
            
            logs[i,0] = tseis
            logs[i,1] = LOGS[i,1]
            logs[i,2] = LOGS[i,2]
            logs[i,3] = dat_out
            
        return logs            
#----------------------------------------------------
if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    tseis = np.arange(0,3601,1) 
    
    data = mfun.load_obj('Poseidon_2')
    
    plt.plot(data.Vs,data.time,'k')   
    
    params = ctrendlogs(data.Vs,data.time,'Vs')  
    #pred = fcompactntrend_vs(data.time,params)
    pred = predict(params,'Vp',data.Vs,tseis)
    plt.plot(pred,data.time,'ro')
    plt.show()



      