import numpy as np

def HotPixelFilter(x,y,t,pol,cam,xdim,ydim, threventhotpixel=100):
    #ignore the Hot Pixel
    #hot pixels are define as the pixels that record a number of event
    #bigger than threventhotpixel
    
    # if len(args)<7:
    #     threventhotpixel= 100; #default value for timehotpixel us

    hotpixelarray=np.zeros([xdim,ydim])
    
    for i in range(0,len(t)):
        hotpixelarray[x[i]-1,y[i]-1]=hotpixelarray[x[i]-1,y[i]-1]+1
    
    
    selindexarray = hotpixelarray>= threventhotpixel
    [hpx,hpy]=np.nonzero(selindexarray)

    for k in range(len(hpx)):
        selindexvector= np.logical_and(x==hpx[k], y==hpy[k])
        x=x[~selindexvector]
        y=y[~selindexvector]
        t=t[~selindexvector]
        pol=pol[~selindexvector]
        cam=cam[~selindexvector]
    
    return x,y,t,pol,cam