# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 07:18:37 2019

@author: Ayodeji Babalola
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matlab_funcs as mfun
#------------------------------------------------------------------------------
def plot_seismic(inputseis,twt,name,colr='seismic',clip_val=None):
    ntraces=np.shape(inputseis)[1]
    if clip_val is None:
        clip_val=abs(np.percentile(inputseis, 0.999))   # default clip value
    f, ax = plt.subplots(figsize=(16,6))
    im=ax.imshow(inputseis,interpolation='bilinear',aspect='auto',cmap=colr,extent=(0,ntraces,twt[-1],twt[0]),vmin=-clip_val,vmax=clip_val)
    plt.xlabel('Trace no.'), plt.ylabel('Two-way time [ms]')
    plt.title(name), plt.grid(), plt.colorbar(im)
    
#------------------------------------------------------------------------------  
def imshow(dat,time = None)  :
    plt.figure()
    if (time is None):
        plt.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto')
        plt.colorbar()
    else:
        pass
       
        

#------------------------------------------------------------------------------  
def imshow_seis(dat,seis_coord = None,well_files = None,horfiles = None)  :
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if (seis_coord is None):
        pos = ax.imshow(dat,interpolation='bicubic',cmap ='seismic', aspect='auto')
        fig.colorbar(pos, cax=cax, orientation='vertical')
    else:        
        #dist = calc_dist(seis_coord)[0]   
        tmin = seis_coord['tseis'][0]
        tmax = seis_coord['tseis'][-1]
        xmin = 0
        xmax = dat.shape[1]*seis_coord['bins'][0]        
        pos = ax.imshow(dat,interpolation='bicubic',cmap = 'seismic', aspect='auto',extent = [xmin,xmax,tmax,tmin])
        
        nwells = well_files.size
        numm = 40
        wellcoord = seis_coord['wellcoord']
        for i in range(nwells):
            wlog = mfun.load_obj(well_files[i])
            ai = wlog.AI
            if (ai is None):
                wlog.calc_AI()
                ai = wlog.AI
            wtmin = np.int(wlog.time[0])
            wtmax = np.int(wlog.time[-1])
            if(wtmin < tmin):
                wtmin = tmin
            elif( wtmax > tmax):
                wtmax = tmax
            zone = np.array([wtmin,wtmax], dtype = int)
            ai = mfun.segment_logs(zone,wlog.time,ai)[0]
            wxmin = xmin + wellcoord['indx'][i]*seis_coord['bins'][0]
            wxmax = wxmin + numm 
            ai_mat = np.tile(ai,[numm,1]).T
            ax.imshow(ai_mat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',extent = [wxmin,wxmax,wtmax,wtmin])
        
        if(horfiles is not None):
            nhor = horfiles.size
            indx = seis_coord['coord_indx']        
            hor_x = np.arange(xmin,xmax,seis_coord['bins'][0]) # temp
            for xd in range(nhor):
                hor = mfun.load_obj(horfiles[xd])
                hor_y = hor.Z[indx]
                ax.plot(hor_x,hor_y,'k')
        
    fig.colorbar(pos, cax=cax, orientation='vertical')
    ax.autoscale() 
    ax.set_xlabel('Distance(m)')
    ax.set_ylabel('Time(ms)')
    ax.set_title('Seismic')        
        
#------------------------------------------------------------------------------  
def imshow_ai(dat,seis_coord = None,well_files = None,horfiles = None)  :
    #maxval = np.max(dat)
    maxval = 17000
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if (seis_coord is None):
        pos = ax.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 4500,vmax =maxval)
        fig.colorbar(pos, cax=cax, orientation='vertical')
    else:        
        #dist = calc_dist(seis_coord)[0]   
        tmin = seis_coord['tseis'][0]
        tmax = seis_coord['tseis'][-1]
        xmin = 0
        xmax = dat.shape[1]*seis_coord['bins'][0]        
        pos = ax.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 4500,vmax =maxval,\
           extent = [xmin,xmax,tmax,tmin])
        
        nwells = well_files.size
        numm = 40
        wellcoord = seis_coord['wellcoord']
        for i in range(nwells):
            wlog = mfun.load_obj(well_files[i])
            ai = wlog.AI
            if (ai is None):
                wlog.calc_AI()
                ai = wlog.AI
            wtmin = np.int(wlog.time[0])
            wtmax = np.int(wlog.time[-1])
            if(wtmin < tmin):
                wtmin = tmin
            elif( wtmax > tmax):
                wtmax = tmax
            zone = np.array([wtmin,wtmax], dtype = int)
            ai = mfun.segment_logs(zone,wlog.time,ai)[0]
            wxmin = xmin + wellcoord['indx'][i]*seis_coord['bins'][0]
            wxmax = wxmin + numm 
            ai_mat = np.tile(ai,[numm,1]).T
            ax.imshow(ai_mat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 4500,vmax =maxval,\
                          extent = [wxmin,wxmax,wtmax,wtmin])
            if(horfiles is not None):
                nhor = horfiles.size
                indx = seis_coord['coord_indx']        
                hor_x = np.arange(xmin,xmax,seis_coord['bins'][0]) # temp
                for xd in range(nhor):
                    hor = mfun.load_obj(horfiles[xd])
                    hor_y = hor.Z[indx]
                    ax.plot(hor_x,hor_y,'k')

    fig.colorbar(pos, cax=cax, orientation='vertical')
    ax.autoscale() 
    ax.set_xlabel('Distance(m)')
    ax.set_ylabel('Time(ms)')
    ax.set_title('AI') 
    
#------------------------------------------------------------------------------  
def imshow_si(dat,seis_coord = None,well_files = None,horfiles = None)  :
    maxval = np.max(dat)
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if (seis_coord is None):
        pos =ax.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 1000,vmax =maxval)
        fig.colorbar(pos, cax=cax, orientation='vertical')
    else:        
        #dist = calc_dist(seis_coord)[0]   
        tmin = seis_coord['tseis'][0]
        tmax = seis_coord['tseis'][-1]
        xmin = 0
        xmax = dat.shape[1]*seis_coord['bins'][0]        
        pos = ax.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 1000,vmax =maxval,\
           extent = [xmin,xmax,tmax,tmin])
        
        nwells = well_files.size
        numm = 40
        wellcoord = seis_coord['wellcoord']
        for i in range(nwells):
            wlog = mfun.load_obj(well_files[i])
            si = wlog.SI
            if (si is None):
                wlog.calc_SI()
                si = wlog.SI
            wtmin = np.int(wlog.time[0])
            wtmax = np.int(wlog.time[-1])
            if(wtmin < tmin):
                wtmin = tmin
            elif( wtmax > tmax):
                wtmax = tmax
            zone = np.array([wtmin,wtmax], dtype = int)
            si = mfun.segment_logs(zone,wlog.time,si)[0]
            wxmin = xmin + wellcoord['indx'][i]*seis_coord['bins'][0]
            wxmax = wxmin + numm 
            ai_mat = np.tile(si,[numm,1]).T
            ax.imshow(ai_mat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 4500,vmax =maxval,\
                          extent = [wxmin,wxmax,wtmax,wtmin])
        if(horfiles is not None):
            nhor = horfiles.size
            indx = seis_coord['coord_indx']        
            hor_x = np.arange(xmin,xmax,seis_coord['bins'][0]) # temp
            for xd in range(nhor):
                hor = mfun.load_obj(horfiles[xd])
                hor_y = hor.Z[indx]
                ax.plot(hor_x,hor_y,'k')

    fig.colorbar(pos, cax=cax, orientation='vertical')
    ax.autoscale() 
    ax.set_xlabel('Distance(m)')
    ax.set_ylabel('Time(ms)')
    ax.set_title('SI')    
#------------------------------------------------------------------------------  
def imshow_rho(dat,seis_coord = None,well_files = None,horfiles = None)  :
    print('BUG : Nan values in log causing issues')
    maxval = np.max(dat)
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)    
    if (seis_coord is None):
        pos =ax.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 2,vmax =maxval)
        fig.colorbar(pos, cax=cax, orientation='vertical')
    else:        
                #dist = calc_dist(seis_coord)[0]   
        tmin = seis_coord['tseis'][0]
        tmax = seis_coord['tseis'][-1]
        xmin = 0
        xmax = dat.shape[1]*seis_coord['bins'][0]        
        pos = ax.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin =2,vmax =maxval,\
           extent = [xmin,xmax,tmax,tmin])
        
        nwells = well_files.size
        numm = 40
        wellcoord = seis_coord['wellcoord']
        for i in range(nwells):
            wlog = mfun.load_obj(well_files[i])
            rho = wlog.Rhob
            if (rho is None):
                raise Exception ('Density log does not exist')
            wtmin = np.int(wlog.time[0])
            wtmax = np.int(wlog.time[-1])
            if(wtmin < tmin):
                wtmin = tmin
            elif( wtmax > tmax):
                wtmax = tmax
            zone = np.array([wtmin,wtmax], dtype = int)
            rho = mfun.segment_logs(zone,wlog.time,rho)[0]
            wxmin = xmin + wellcoord['indx'][i]*seis_coord['bins'][0]
            wxmax = wxmin + numm 
            ai_mat = np.tile(rho,[numm,1]).T
            ax.imshow(ai_mat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 4500,vmax =maxval,\
                          extent = [wxmin,wxmax,wtmax,wtmin])
                          
        if(horfiles is not None):
            nhor = horfiles.size
            indx = seis_coord['coord_indx']        
            hor_x = np.arange(xmin,xmax,seis_coord['bins'][0]) # temp
            for xd in range(nhor):
                hor = mfun.load_obj(horfiles[xd])
                hor_y = hor.Z[indx]
                ax.plot(hor_x,hor_y,'k')

    fig.colorbar(pos, cax=cax, orientation='vertical')
    ax.autoscale() 
    ax.set_xlabel('Distance(m)')
    ax.set_ylabel('Time(ms)')
    ax.set_title('RHOB') 
    
def imshow_sw(dat,seis_coord = None,well_files = None,horfiles = None,rockprop = None)  :
    minval = 0
    maxval = np.max(dat)
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)    
    if (seis_coord is None):
        pos =ax.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 0,vmax =maxval)
        fig.colorbar(pos, cax=cax, orientation='vertical')
    else:        
                #dist = calc_dist(seis_coord)[0]   
        tmin = seis_coord['tseis'][0]
        tmax = seis_coord['tseis'][-1]
        xmin = 0
        xmax = dat.shape[1]*seis_coord['bins'][0]        
        pos = ax.imshow(dat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin =0,vmax =maxval,\
           extent = [xmin,xmax,tmax,tmin])
        
        nwells = well_files.size
        numm = 40
        wellcoord = seis_coord['wellcoord']
        for i in range(nwells):
            wlog = mfun.load_obj(well_files[i])
            rho = wlog.Rhob
            if (rho is None):
                raise Exception ('Density log does not exist')
            wtmin = np.int(wlog.time[0])
            wtmax = np.int(wlog.time[-1])
            if(wtmin < tmin):
                wtmin = tmin
            elif( wtmax > tmax):
                wtmax = tmax
            zone = np.array([wtmin,wtmax], dtype = int)
            rho = mfun.segment_logs(zone,wlog.time,rho)[0]
            wxmin = xmin + wellcoord['indx'][i]*seis_coord['bins'][0]
            wxmax = wxmin + numm 
            ai_mat = np.tile(rho,[numm,1]).T
            ax.imshow(ai_mat,interpolation='bicubic',cmap = 'nipy_spectral', aspect='auto',vmin = 0,vmax =maxval,\
                          extent = [wxmin,wxmax,wtmax,wtmin])
                          
        if(horfiles is not None):
            nhor = horfiles.size
            indx = seis_coord['coord_indx']        
            hor_x = np.arange(xmin,xmax,seis_coord['bins'][0]) # temp
            for xd in range(nhor):
                hor = mfun.load_obj(horfiles[xd])
                hor_y = hor.Z[indx]
                ax.plot(hor_x,hor_y,'k')
    
    fig.colorbar(pos, cax=cax, orientation='vertical')
    ax.autoscale() 
    ax.set_xlabel('Distance(m)')
    ax.set_ylabel('Time(ms)')
    if (rockprop is None):
        rockprop = 'Porosity'
    ax.set_title(rockprop)     
#------------------------------------------------------------------------------    
def calc_dist(seis_coord):
    min_x = np.min(seis_coord['X'])
    min_y = np.min(seis_coord['Y'])
    dist = np.sqrt(((seis_coord['X'] - min_x)**2 + (seis_coord['Y'] - min_y)**2))
    indx  = np.argsort(dist)
    return dist,indx.astype(int)  
    