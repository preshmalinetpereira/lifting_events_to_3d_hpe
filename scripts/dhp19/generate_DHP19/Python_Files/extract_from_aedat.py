import numpy as np
from HotPixelFilter import HotPixelFilter 
from BackgroundFilter import BackgroundFilter
from maskRegionFilter import maskRegionFilter
import functools
def extract_from_aedat(aedat, events, startTime, stopTime, sx, sy, nbcam, thrEventHotPixel, dt, xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2):
    if events[-1] > events[0]:
        # startIndex: event right after startTime
        # stopIndex:  event right before stopTime
        startIndexes = np.where(events > startTime)[0]
        stopIndexes = np.where(events < stopTime)[0]
        startIndex = startIndexes[0] -1
        stopIndex  = stopIndexes[-1]
    else:
        # special case if timeStamp overflows(for S14_1_1) 
        startIndexes = np.where(events == startTime)[0]
        stopIndexes = np.where(events == stopTime)[0]
        startIndex = startIndexes[0] -1
        stopIndex  = stopIndexes[0]
    
    
    pol       = aedat['data']['polarity']['polarity'][startIndex:stopIndex]
    x         = aedat['data']['polarity']['x'][startIndex:stopIndex]
    y_raw     = aedat['data']['polarity']['y'][startIndex:stopIndex]
    cam       = aedat['data']['polarity']['cam'][startIndex:stopIndex]
    timeStamp = events[startIndex:stopIndex]
    
    # remove events out of the boundaries.
    cond = functools.reduce(np.logical_or, ([x<0] , [x>sx-1] , [y_raw<0] , [y_raw>sy-1] , [cam<0] , [cam>nbcam-1]))[0]
    x = x[~cond]
    y_raw = y_raw[~cond]
    pol = pol[~cond]
    cam = cam[~cond]
    timeStamp = timeStamp[~cond]
    
    X = (sx-x)+cam*sx
    y = sy-y_raw
    
    # apply filters to the events.
    X_tmp, y_tmp, timeStamp_tmp, pol_tmp, cam_tmp = HotPixelFilter(X, y, timeStamp, pol, cam, sx*nbcam, sy, thrEventHotPixel)
    
    X_tmp1, y_tmp1, timeStamp_tmp1, pol_tmp1, cam_tmp1 = BackgroundFilter(X_tmp, y_tmp, 
        timeStamp_tmp, pol_tmp, cam_tmp, sx*nbcam, sy, dt)
    #X_tmp1, y_tmp1, timeStamp_tmp1, pol_tmp1, cam_tmp1 = BackgroundFilter(X, y,timeStamp, pol, cam, sx*nbcam, sy, dt)
    
    X_tmp2, y_tmp2, timeStamp_tmp2, pol_tmp2, cam_tmp2 = maskRegionFilter(X_tmp1, y_tmp1, 
        timeStamp_tmp1, pol_tmp1, cam_tmp1, xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1)
    
    X_tmp3, y_tmp3, timeStamp_tmp3, pol_tmp3, cam_tmp3 = maskRegionFilter(X_tmp2, y_tmp2, 
        timeStamp_tmp2, pol_tmp2, cam_tmp2, xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2)

    return startIndex, stopIndex, pol_tmp3, X_tmp3, y_tmp3, cam_tmp3, timeStamp_tmp3
