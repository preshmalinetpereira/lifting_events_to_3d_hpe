import numpy as np
import math

def BackgroundFilter(x,y,t,pol,cam,xdim,ydim,dt):
    #filter out the events that are not support the neighborhood events
    #dt, define the time to consider an event valid or not
    #if nargin<7
    #    dt= 30000 #default value for dt us
    #end

    lastTimesMap=np.zeros(xdim,ydim)
    index=np.zeros(len(t),1)
    for i in range(1,len(t)):
        ts=t[i], xs=x[i], ys=y[i]
        deltaT=ts-lastTimesMap[xs,ys]
        if deltaT>dt:
            index[i]=math.nan
    
    
        if not (xs==1 or xs==xdim or ys==1 or ys==ydim):
            lastTimesMap[xs-1, ys] = ts
            lastTimesMap[xs+1, ys] = ts
            lastTimesMap[xs, ys-1] = ts
            lastTimesMap[xs, ys+1] = ts
            lastTimesMap[xs-1, ys-1] = ts
            lastTimesMap[xs+1, ys+1] = ts
            lastTimesMap[xs-1, ys+1] = ts
            lastTimesMap[xs+1, ys-1] = ts
        
    return x,y,t,pol,cam