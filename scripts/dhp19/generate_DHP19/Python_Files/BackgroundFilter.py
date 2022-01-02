### File created by preshma
import numpy as np
import math

def BackgroundFilter(x,y,t,pol,cam,xdim,ydim,dt):
    lastTimesMap=np.zeros((xdim,ydim))
    index=np.zeros((len(t),1))
    for i in range(len(t)):
        ts=t[i]
        xs=x[i]-1 #to compensate for matlab indexing starting from 1
        ys=y[i]-1
        deltaT=ts-lastTimesMap[xs,ys] 
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

    x=x[np.transpose(~(np.isnan(index)))[0]]
    y=y[np.transpose(~(np.isnan(index)))[0]]
    t=t[np.transpose(~(np.isnan(index)))[0]]
    pol=pol[np.transpose(~(np.isnan(index)))[0]]
    cam=cam[np.transpose(~(np.isnan(index)))[0]]

    return x,y,t,pol,cam