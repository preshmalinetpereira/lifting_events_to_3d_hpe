import numpy as np
import math

def BackgroundFilter(x,y,t,pol,cam,xdim,ydim,dt):
    #filter out the events that are not support the neighborhood events
    #dt, define the time to consider an event valid or not
    #if nargin<7
    #    dt= 30000 #default value for dt us
    #end

    lastTimesMap=np.zeros((xdim,ydim))
    index=np.zeros((len(t),1))
    for i in range(len(t)):
        ts=t[i]
        xs=x[i]-1
        ys=y[i]-1
        deltaT=ts-lastTimesMap[xs,ys] #change to xs-1 as the value of x is as per matlab
        if deltaT>dt:
            index[i]=math.nan
    
    
        if not (xs==0 or xs==xdim-1 or ys==0 or ys==ydim-1):
            lastTimesMap[xs-1, ys] = ts
            lastTimesMap[xs+1, ys] = ts
            lastTimesMap[xs, ys-1] = ts
            lastTimesMap[xs, ys+1] = ts
            lastTimesMap[xs-1, ys-1] = ts
            lastTimesMap[xs+1, ys+1] = ts
            lastTimesMap[xs-1, ys+1] = ts
            lastTimesMap[xs+1, ys-1] = ts

    x=x[np.transpose(np.logical_not(np.isnan(index)))[0]]
    y=y[np.transpose(np.logical_not(np.isnan(index)))[0]]
    t=t[np.transpose(np.logical_not(np.isnan(index)))[0]]
    pol=pol[np.transpose(np.logical_not(np.isnan(index)))[0]]
    cam=cam[np.transpose(np.logical_not(np.isnan(index)))[0]]

    return x,y,t,pol,cam