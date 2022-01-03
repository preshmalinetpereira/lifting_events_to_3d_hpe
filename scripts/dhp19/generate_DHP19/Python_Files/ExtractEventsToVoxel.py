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
    img = np.zeros((sx*nbcam+1, sy+1))
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
    dt = float(timeStamp[init_slice+eventsPerFullFrame] - t0)

    for idx in range(len(timeStamp)):

        coordx = X[idx]
        coordy = y[idx]
        pi = pol[idx]
        ti = timeStamp[idx]
        
        # Constant event count accumulation.
        counter = counter + 1
        img[coordx,coordy, 1] = img[coordx,coordy,1] + 1
        t = float(B -1 ) / dt * float(ti - t0) + 1

        for tn in range(B):
            voxel[coordx,coordy, tn] = voxel[coordx,coordy, tn] +  pi * max(0, 1 - abs(tn - t))
        
        if (counter >= eventsPerFullFrame):
            init_slice = idx+1
            final_slice = min(init_slice+eventsPerFullFrame, len(timeStamp))
            t0 = timeStamp[init_slice]
            dt = float(timeStamp[final_slice] - t0)

            # k is the time duration (in ms) of the recording up until the
            # current finished accumulated frame.
            k = np.floor((timeStamp[idx] - startTime)*0.0001)+1

            # if k is larger than the label at the end of frame
            # accumulation, the generation of frames stops.

            if k > len(XYZPOS['XYZPOS']['head'][0][0]):
                break
            img = np.delete(np.delete(img, -1,-1), 0, 0)
            # arrange image in channels.
            I1=img[0:sx,:]
            I2=img[sx:2*sx,:]
            I3=img[2*sx:3*sx,:]
            I4=img[3*sx:4*sx,:]

            # arrange image in channels.
            v1=voxel[0:sx,:,:]
            v2=voxel[sx:2*sx,:,:]
            v3=voxel[2*sx:3*sx,:,:]
            v4=voxel[3*sx:4*sx,:,:]

            # subsampling
            if do_subsampling:
              I1s = subsample(I1,sx,sy,reshapex,reshapey, 'center')
              # different crop location as data is shifted to right side.
              I2s = subsample(I2,sx,sy,reshapex,reshapey, 'begin')
              I3s = subsample(I3,sx,sy,reshapex,reshapey, 'center')
              I4s = subsample(I4,sx,sy,reshapex,reshapey, 'center') 

              v1 = subsample(v1,sx,sy,reshapex,reshapey, 'center')
              #different crop location as data is shifted to right side.
              v2 = subsample(v2,sx,sy,reshapex,reshapey, 'begin')
              v3 = subsample(v3,sx,sy,reshapex,reshapey, 'center')
              v4 = subsample(v4,sx,sy,reshapex,reshapey, 'center')
            else:
              I1s = I1
              I2s = I2
              I3s = I3
              I4s = I4
            # end

            # Normalization
            I1n = normalizeImage3Sigma(I1s) 
            I2n = normalizeImage3Sigma(I2s) 
            I3n = normalizeImage3Sigma(I3s) 
            I4n = normalizeImage3Sigma(I4s) 

            V1n = (v1)
            V2n = (v2) 
            V3n = (v3) 
            V4n = (v4) 

            with open('%s_frame_%s_cam_0_timesurface.mat'%(fileName, nbFrame-1), 'wb') as file: pickle.dump(V1n, file)
            with open('%s_frame_%s_cam_1_timesurface.mat'%(fileName, nbFrame-1), 'wb') as file: pickle.dump(V2n, file)
            with open('%s_frame_%s_cam_2_timesurface.mat'%(fileName, nbFrame-1), 'wb') as file: pickle.dump(V3n, file)
            with open('%s_frame_%s_cam_3_timesurface.mat'%(fileName, nbFrame-1), 'wb') as file: pickle.dump(V4n, file)

            VoxelMovie[1,:,:,:,nbFrame] = V1n
            VoxelMovie[2,:,:,:,nbFrame] = V2n
            VoxelMovie[3,:,:,:,nbFrame] = V3n
            VoxelMovie[4,:,:,:,nbFrame] = V4n

            last_k = k
            counter = 0
            img = np.zeros(sx*nbcam+1,sy+1)
            voxel = np.zeros(sx*nbcam, sy, B)
            nbFrame = nbFrame + 1

    print('Number of frame: ' + str(nbFrame))
    fileID.write('%s \t frames: %d\n'%(fileName, nbFrame)) 
  



