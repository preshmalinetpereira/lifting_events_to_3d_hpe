### File created by preshma
import numpy as np

def HotPixelFilter(x,y,t,pol,cam,xdim,ydim, threventhotpixel=100):
    #ignore the Hot Pixel
    #hot pixels are define as the pixels that record a number of event
    #bigger than threventhotpixel
    
    # if len(args)<7:
    #     threventhotpixel= 100; #default value for timehotpixel us

    hotpixelarray=np.zeros([xdim+1,ydim+1])
    
    for i in range(len(t)):
        hotpixelarray[x[i],y[i]]=hotpixelarray[x[i],y[i]]+1 #allowed the program to start array from 1 to avoid additional computation time as this is an intermediate array
              
    selindexarray = hotpixelarray>= threventhotpixel
    [hpx,hpy]=np.nonzero(selindexarray.astype(int))

    for k in range(len(hpx)):
        selindexvector= np.logical_and(x==hpx[k], y==hpy[k])
        x=x[~(selindexvector)]
        y=y[~(selindexvector)]
        t=t[~(selindexvector)]
        pol=pol[~(selindexvector)]
        cam=cam[~(selindexvector)]
    
    return x,y,t,pol,cam