import numpy as np

def HotPixelFilter(*args):
    #ignore the Hot Pixel
    #hot pixels are define as the pixels that record a number of event
    #bigger than threventhotpixel
    
    if len(args)<7:
        threventhotpixel= 100; #default value for timehotpixel us

    x=args[0],y = args[1],t= args[2],pol= args[3],cam= args[4],xdim= args[5],ydim= args[6],threventhotpixel= args[7]
    hotpixelarray=np.zeros(xdim,ydim)
    
    for i in range(1,len(t)):
        hotpixelarray[x[i],y[i]]=hotpixelarray[x[i],y[i]]+1
    
    
    selindexarray = hotpixelarray>= threventhotpixel
    [hpx,hpy]=np.nonzero(selindexarray)

    for k in range(1, len(hpx)):
        selindexvector= x==hpx[k] & y==hpy(k)
        x=x(not selindexvector)
        y=y(not selindexvector)
        t=t(not selindexvector)
        pol=pol(not selindexvector)
        cam=cam(not selindexvector)
    
    return x,y,t,pol,cam