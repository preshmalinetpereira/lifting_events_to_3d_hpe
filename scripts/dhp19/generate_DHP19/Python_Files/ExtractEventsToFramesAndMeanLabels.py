import numpy as np
from extract_from_aedat import extract_from_aedat
from subsample import subsample
from normalizeImage3Sigma import normalizeImage3Sigma
import h5py
import os

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
    startIndex, stopIndex, pol, X, y, cam, timeStamp = extract_from_aedat( aedat, events, 
                        startTime, stopTime, sx, sy, nbcam, thrEventHotPixel, dt, 
                        xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, 
                        xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2)

    # Initialization
    nbFrame_initialization = round(len(timeStamp)/eventsPerFullFrame)
    img = np.zeros((sx*nbcam, sy))
    pose = np.zeros((13, 3))
    temp = np.empty([nbcam, reshapex, reshapey, nbFrame_initialization])
    IMovie = temp.fill(np.nan)
    temp = np.empty([13, 3, nbFrame_initialization])
    poseMovie = temp.fill(np.nan)

    last_k = 1
    counter = 0
    nbFrame = 0
    

    countPerFrame = eventsPerFullFrame

    
    for idx in range(0,len(timeStamp)):

        coordx = X(idx)
        coordy = y(idx)
        
        # Constant event count accumulation.
        counter = counter + 1
        img[coordx,coordy] = img[coordx,coordy] + 1

        if (counter >= countPerFrame):
            nbFrame = nbFrame + 1
            # k is the time duration (in ms) of the recording up until the
            # current finished accumulated frame.
            k = np.floor((timeStamp(idx) - startTime)*0.0001)+1
            
            # if k is larger than the label at the end of frame
            # accumulation, the generation of frames stops.
            if k > len(XYZPOS.XYZPOS.head):
                break
            
            
            # arrange image in channels.
            I1=img[0:sx-1,:]
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
            I1n = normalizeImage3Sigma(I1s) #uint8(normalizeImage3Sigma(I1s))
            I2n = normalizeImage3Sigma(I2s) #uint8(normalizeImage3Sigma(I2s))
            I3n = normalizeImage3Sigma(I3s) # uint8(normalizeImage3Sigma(I3s))
            I4n = normalizeImage3Sigma(I4s) #uint8(normalizeImage3Sigma(I4s)) 

            # #
            IMovie[1,:,:,nbFrame] = I1n
            IMovie[2,:,:,nbFrame] = I2n
            IMovie[3,:,:,nbFrame] = I3n
            IMovie[4,:,:,nbFrame] = I4n
            
            # #
            pose[1,:] = np.nanmean(XYZPOS[XYZPOS]['head'][last_k:k,:],0)
            pose[2,:] = np.nanmean(XYZPOS[XYZPOS]['shoulderR'][last_k:k,:],0)
            pose[3,:] = np.nanmean(XYZPOS[XYZPOS]['shoulderL'][last_k:k,:],0)
            pose[4,:] = np.nanmean(XYZPOS[XYZPOS]['elbowR'][last_k:k,:],0)
            pose[5,:] = np.nanmean(XYZPOS[XYZPOS]['elbowL'][last_k:k,:],0)
            pose[6,:] = np.nanmean(XYZPOS[XYZPOS]['hipR'][last_k:k,:],0)
            pose[7,:] = np.nanmean(XYZPOS[XYZPOS]['hipL'][last_k:k,:],0)
            pose[8,:] = np.nanmean(XYZPOS[XYZPOS]['handR'][last_k:k,:],0)
            pose[9,:] = np.nanmean(XYZPOS[XYZPOS]['handL'][last_k:k,:],0)
            pose[10,:] = np.nanmean(XYZPOS[XYZPOS]['kneeR'][last_k:k,:],0)
            pose[11,:] = np.nanmean(XYZPOS[XYZPOS]['kneeL'][last_k:k,:],0)
            pose[12,:] = np.nanmean(XYZPOS[XYZPOS]['footR'][last_k:k,:],0)
            pose[13,:] = np.nanmean(XYZPOS[XYZPOS]['footL'][last_k:k,:],0)
            
            poseMovie[:,:,nbFrame] = pose
            
            last_k = k
            # dt = timeStamp(idx) - lastTimeStampLastFrame
            # lastTimeStampLastFrame = timeStamp(idx)
            
            # initialize for next frame.
            # figure(1)
            # imshow(I2s)
            counter = 0
            img = np.zeros((sx*nbcam,sy))

    print('Number of frame: ' + str(nbFrame))
    fileID.write('%s \t frames: %d\n'%(fileName, nbFrame)) 
    
    if saveHDF5 == 1:
        DVSfilenameh5 = fileName + '.h5'
        IMovie = IMovie[:,:,:,1:nbFrame]
        
        if convert_labels == True:
            Labelsfilenameh5 = fileName + '_label.h5'
            poseMovie = poseMovie[:,:,1:nbFrame]
        
        if os.path.isfile(DVSfilenameh5):
            return
        else:
            f = h5py.File('/DVS/' + str(DVSfilenameh5), 'w')
            f.create_dataset(DVSfilenameh5,IMovie.astype(np.uint8))
            # h5create(DVSfilenameh5,'/DVS',[nbcam reshapex reshapey nbFrame]);
            # h5write(DVSfilenameh5, '/DVS', uint8(IMovie)); 
            if convert_labels == True:
              f = h5py.File('/XYZ/' + str(Labelsfilenameh5), 'w')
              f.create_dataset(Labelsfilenameh5,poseMovie)
                #h5create(Labelsfilenameh5,'/XYZ',[13 3 nbFrame])
                #h5write(Labelsfilenameh5,'/XYZ',poseMovie)