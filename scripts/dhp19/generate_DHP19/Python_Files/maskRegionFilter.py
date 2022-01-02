import functools
import numpy as np
def maskRegionFilter(x,y,t,pol,cam,xmin,xmax,ymin,ymax):
   cond = functools.reduce(np.logical_and, ([x>xmin] , [x<xmax], [y>ymin] , [y<ymax]))[0]
   #cond=np.logical_and([x>xmin] , [x<xmax], [y>ymin] , [y<ymax])
   x2=x[np.logical_not(cond)]
   y2=y[np.logical_not(cond)]
   t2=t[np.logical_not(cond)]
   pol2=pol[np.logical_not(cond)]
   cam2=cam[np.logical_not(cond)]
   return x2, y2, t2, pol2, cam2
   