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


def ExtractEventsToTimeSurface(
            fileID, # log file
            aedat, events, eventsPerFullFrame, 
            startTime, stopTime, fileName, 
            XYZPOS, sx, sy, nbcam, thrEventHotPixel, dt, 
            xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, # 1st mask coordinates
            xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2, # 2nd mask coordinates
            do_subsampling, reshapex, reshapey, 
            saveHDF5, convert_labels):

    save_count_frames = False
    startTime = startTime.astype(np.int32)
    stopTime  = stopTime.astype(np.int32)
    
    # # Extract and filter events from aedat
    # startIndex, stopIndex, pol, X, y, cam, timeStamp = extract_from_aedat( aedat, events, startTime, stopTime, sx, sy, nbcam, thrEventHotPixel, dt, 
    #                     xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, 
    #                     xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2)

    file = open('extractfromaedat_data.txt', 'rb')
    d = pickle.load(file)
    startIndex, stopIndex, pol, X, y, cam, timeStamp = d['startIndex'], d['stopIndex'], d['pol'], d['X'], d['y'], d['cam'], d['timeStamp'] #For faster execution, loaded data from file
  
    # Initialization
    nbFrame_initialization = round(len(timeStamp)/eventsPerFullFrame)
    acc = np.zeros((sx*nbcam+1, sy+1, 2+1))
    save_output= True
    counter = 0
    nbFrame = -1
    delta = 300000

    plt.ion()
    fig = plt.figure()

    init_slice = 1
    t0 = timeStamp[init_slice]
    
    for idx in range(len(timeStamp)):

        coordx = X[idx]
        coordy = y[idx]
        pi = pol[idx]
        ti = timeStamp[idx]
        acc[coordx, coordy, pi] = ti
        
        # Constant event count accumulation.
        counter = counter + 1

        if (counter >= eventsPerFullFrame):
            t0 = timeStamp[idx]
            img = np.exp(-(float(t0) - acc) / delta)
            # k is the time duration (in ms) of the recording up until the
            # current finished accumulated frame.
            k = int(np.floor((timeStamp[idx] - startTime)*0.0001)+1)
            
            # if k is larger than the label at the end of frame accumulation
            # the generation of frames stops.
            if k > len(XYZPOS['XYZPOS']['head'][0][0]):
                break
            
            # arrange image in channels.
            I1=img[0:sx,:]
            I2=img[sx:2*sx,:]
            I3=img[2*sx:3*sx,:]
            I4=img[3*sx:4*sx,:]
              
            # Normalization
            V1n = normalizeImage3Sigma(I1[:, :, 0] - I1[:, :, 1])
            V2n = normalizeImage3Sigma(I2[:, :, 0] - I2[:, :, 1]) 
            V3n = normalizeImage3Sigma(I3[:, :, 0] - I3[:, :, 1]) 
            V4n = normalizeImage3Sigma(I4[:, :, 0] - I4[:, :, 1]) 


            
            if save_output:
                with open('%s_frame_%s_cam_0_timesurface.mat'%(fileName, nbFrame-1), 'wb') as file: pickle.dump(V1n, file)
                with open('%s_frame_%s_cam_1_timesurface.mat'%(fileName, nbFrame-1), 'wb') as file: pickle.dump(V2n, file)
                with open('%s_frame_%s_cam_2_timesurface.mat'%(fileName, nbFrame-1), 'wb') as file: pickle.dump(V3n, file)
                with open('%s_frame_%s_cam_3_timesurface.mat'%(fileName, nbFrame-1), 'wb') as file: pickle.dump(V4n, file)
                    
            
            nbFrame = nbFrame+1
            plt.imshow(V2n)
            plt.show()
            plt.pause(10**-14)
            counter = 0

    print('Number of frame: ' + str(nbFrame))
    fileID.write('%s \t frames: %d\n'%(fileName, nbFrame)) 
