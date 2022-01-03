### File created by preshma
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from extract_from_aedat import extract_from_aedat
from normalizeImage3Sigma import normalizeImage3Sigma
from subsample import subsample


def ExtractEventsToFramesAndMeanLabels(
            fileID, # log file
            aedat, events, eventsPerFullFrame, 
            startTime, stopTime, fileName, 
            XYZPOS, sx, sy, nbcam, thrEventHotPixel, dt, 
            xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, # 1st mask coordinates
            xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2, # 2nd mask coordinates
            do_subsampling, reshapex, reshapey, 
            saveHDF5, convert_labels):

    startTime = startTime.astype(np.int32)
    stopTime  = stopTime.astype(np.int32)
    
    # Extract and filter events from aedat
    startIndex, stopIndex, pol, X, y, cam, timeStamp = extract_from_aedat( aedat, events, startTime, stopTime, sx, sy, nbcam, thrEventHotPixel, dt, 
                        xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, 
                        xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2)
    # # data = (startIndex, stopIndex, pol, X, y, cam, timeStamp) #For faster execution, dump data to file
    # # with open('extractfromaedat_data.txt', 'wb') as file:
    # #   pickle.dump(data, file)

    # file = open('extractfromaedat_data.txt', 'rb')
    # d = pickle.load(file)
    # startIndex, stopIndex, pol, X, y, cam, timeStamp = d['startIndex'], d['stopIndex'], d['pol'], d['X'], d['y'], d['cam'], d['timeStamp'] #For faster execution, loaded data from file
  

    # Initialization
    nbFrame_initialization = round(len(timeStamp)/eventsPerFullFrame)
    img = np.zeros((sx*nbcam+1, sy+1))
    pose = np.zeros((13, 3))
    IMovie = np.empty([nbcam, reshapex, reshapey, nbFrame_initialization])
    IMovie.fill(np.nan)
    poseMovie = np.empty([13, 3, nbFrame_initialization])
    poseMovie.fill(np.nan)

    last_k = 0
    counter = 0
    nbFrame = -1
    plt.ion()
    fig = plt.figure()
    countPerFrame = eventsPerFullFrame

    
    for idx in range(len(timeStamp)):

        coordx = X[idx]
        coordy = y[idx]
        
        # Constant event count accumulation.
        counter = counter + 1
        img[coordx,coordy] = img[coordx,coordy] + 1

        if (counter >= countPerFrame):
            nbFrame = nbFrame + 1
            # k is the time duration (in ms) of the recording up until the
            # current finished accumulated frame.
            k = int(np.floor((timeStamp[idx] - startTime)*0.0001)+1)
            
            # if k is larger than the label at the end of frame accumulation
            # the generation of frames stops.
            if k > len(XYZPOS['XYZPOS']['head'][0][0]):
                break
            
            img = np.delete(np.delete(img, -1,-1), 0, 0)
            # arrange image in channels.
            I1=img[0:sx,:]
            I2=img[sx:2*sx,:]
            I3=img[2*sx:3*sx,:]
            I4=img[3*sx:4*sx,:]
            
            # subsampling
            if do_subsampling:
              I1s = subsample(I1,sx,sy,reshapex,reshapey, 'center')
              # different crop location as data is shifted to right side.
              I2s = subsample(I2,sx,sy,reshapex,reshapey, 'begin')
              I3s = subsample(I3,sx,sy,reshapex,reshapey, 'center')
              I4s = subsample(I4,sx,sy,reshapex,reshapey, 'center') 
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

            # #
            IMovie[0,:,:,nbFrame] = I1n
            IMovie[1,:,:,nbFrame] = I2n
            IMovie[2,:,:,nbFrame] = I3n
            IMovie[3,:,:,nbFrame] = I4n
            
            # #
            pose[0,:] = np.nanmean(XYZPOS['XYZPOS']['head'][0][0][last_k:k,:],0)
            pose[1,:] = np.nanmean(XYZPOS['XYZPOS']['shoulderR'][0][0][last_k:k,:],0)
            pose[2,:] = np.nanmean(XYZPOS['XYZPOS']['shoulderL'][0][0][last_k:k,:],0)
            pose[3,:] = np.nanmean(XYZPOS['XYZPOS']['elbowR'][0][0][last_k:k,:],0)
            pose[4,:] = np.nanmean(XYZPOS['XYZPOS']['elbowL'][0][0][last_k:k,:],0)
            pose[5,:] = np.nanmean(XYZPOS['XYZPOS']['hipR'][0][0][last_k:k,:],0)
            pose[6,:] = np.nanmean(XYZPOS['XYZPOS']['hipL'][0][0][last_k:k,:],0)
            pose[7,:] = np.nanmean(XYZPOS['XYZPOS']['handR'][0][0][last_k:k,:],0)
            pose[8,:] = np.nanmean(XYZPOS['XYZPOS']['handL'][0][0][last_k:k,:],0)
            pose[9,:] = np.nanmean(XYZPOS['XYZPOS']['kneeR'][0][0][last_k:k,:],0)
            pose[10,:] = np.nanmean(XYZPOS['XYZPOS']['kneeL'][0][0][last_k:k,:],0)
            pose[11,:] = np.nanmean(XYZPOS['XYZPOS']['footR'][0][0][last_k:k,:],0)
            pose[12,:] = np.nanmean(XYZPOS['XYZPOS']['footL'][0][0][last_k:k,:],0)
            
            poseMovie[:,:,nbFrame] = pose
            
            last_k = k
            
            # initialize for next frame.            
            plt.imshow(I2s)
            plt.show()
            plt.pause(0.000000001)
            counter = 0
            img = np.zeros([sx*nbcam+1,sy+1])

    print('Number of frame: ' + str(nbFrame))
    fileID.write('%s \t frames: %d\n'%(fileName, nbFrame)) 
    
    if saveHDF5 == 1:
        DVSfilenameh5 = fileName + '.h5'
        IMovie = IMovie[:,:,:,0:nbFrame]
        
        if convert_labels == True:
            Labelsfilenameh5 = fileName + '_label.h5'
            poseMovie = poseMovie[:,:,0:nbFrame]
        
        if os.path.isfile(DVSfilenameh5):
            return
        else:
            f = h5py.File(str(DVSfilenameh5), 'w')
            f.create_dataset(DVSfilenameh5,data = IMovie, dtype='i8')
            if convert_labels == True:
              f = h5py.File(str(Labelsfilenameh5), 'w')
              f.create_dataset(Labelsfilenameh5,data =poseMovie, dtype='i8')
