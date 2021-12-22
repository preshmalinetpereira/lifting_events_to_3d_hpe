def maskRegionFilter(x,y,t,pol,cam,xmin,xmax,ymin,ymax):
   cond=(x>xmin) and (x<xmax) and (y>ymin) and (y<ymax)
   x2=x(not cond)
   y2=y(not cond)
   t2=t(not cond)
   pol2=pol(not cond)
   cam2=cam(not cond)
   return x2, y2, t2, pol2, cam2
   