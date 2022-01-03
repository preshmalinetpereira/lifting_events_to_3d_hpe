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

    if os.path.isfile(fileName +'_voxel.h5'):
        return

    save_count_frames = False
    startTime = startTime.astype(np.int32)
    stopTime  = stopTime.astype(np.int32)

    # Extract and filter events from aedat
    startIndex, stopIndex, pol, X, y, cam, timeStamp = extract_from_aedat( aedat, events, startTime, stopTime, sx, sy, nbcam, thrEventHotPixel, dt, 
                    xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, 
                    xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2)

    # Initialization
    B = 4
    nbFrame_initialization = round(len(timeStamp)/eventsPerFullFrame)
    img = np.zeros((sx*nbcam, sy))
    voxel = np.zeros((sx*nbcam, sy, B))
    pose = np.zeros((13, 3))

    IMovie = np.empty((nbcam, reshapex, reshapey, nbFrame_initialization))
    IMovie.fill(np.nan)
    VoxelMovie = np.empty((nbcam, reshapex, reshapey, B, nbFrame_initialization))
    VoxelMovie.fill(np.nan)
    poseMovie = np.empty([13, 3, nbFrame_initialization])
    poseMovie.fill(np.nan)

    last_k = 0
    counter = 0
    nbFrame = 0
    plt.ion()
    fig = plt.figure()
    countPerFrame = eventsPerFullFrame

    init_slice = 1
    t0 = timeStamp[init_slice]
    dt = float(timeStamp(init_slice+eventsPerFullFrame) - t0)