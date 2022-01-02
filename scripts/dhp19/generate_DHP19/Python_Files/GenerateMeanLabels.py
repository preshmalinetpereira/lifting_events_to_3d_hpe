### File created by preshma
import numpy as np
from extract_from_aedat import extract_from_aedat
from subsample import subsample
from normalizeImage3Sigma import normalizeImage3Sigma
import h5py
import os
import pickle
import matplotlib.pyplot as plt


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

        # Constant event count accumulation.
        counter = counter + 1

        if (counter >= countPerFrame):
            nbFrame = nbFrame + 1
            # k is the time duration (in ms) of the recording up until the
            # current finished accumulated frame.
            k = int(np.floor((timeStamp[idx] - startTime)*0.0001)+1)
            
            # if k is larger than the label at the end of frame accumulation
            # the generation of frames stops.
            if k > len(XYZPOS['XYZPOS']['head'][0][0]):
                break
           
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

            counter = 0

    print('Number of frame: ' + str(nbFrame))
    fileID.write('%s \t frames: %d\n'%(fileName, nbFrame)) 
    
  
    Labelsfilenameh5 = fileName + '_label.h5'
    poseMovie = poseMovie[:,:,0:nbFrame]

    if os.path.isfile(DVSfilenameh5):
        return
    else:
        f = h5py.File(str(Labelsfilenameh5), 'w')
        f.create_dataset(Labelsfilenameh5,data =poseMovie, dtype='i8')