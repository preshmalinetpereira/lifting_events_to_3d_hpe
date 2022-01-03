### File created by preshma
import math
import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from extract_from_aedat import extract_from_aedat
from normalizeImage3Sigma import normalizeImage3Sigma
from subsample import subsample


def ExtractEventsToVoxel(
            fileID, # log file
            aedat, events, eventsPerFullFrame, 
            startTime, stopTime, fileName, 
            XYZPOS, sx, sy, nbcam, thrEventHotPixel, dt, 
            xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, # 1st mask coordinates
            xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2, # 2nd mask coordinates
            do_subsampling, reshapex, reshapey, 
            saveHDF5, convert_labels):