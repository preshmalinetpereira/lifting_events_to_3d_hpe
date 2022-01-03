#--------------------------------------------------------------------------
# Use this script to generate the accumulated frames of DHP19 dataset.
# The script loops over all the DVS recordings and generates .h5 files
# of constant count frames.
# Currently, only constant count frame generation is implemented/tested.
#
# To import the aedat files here we use a modified version of
# ImportAedatDataVersion1or2, to account for the camera index originating
# each event.
#--------------------------------------------------------------------------

### File created by preshma


from datetime import datetime
from ExtractEventsToTimeSurface import ExtractEventsToVoxelAndMeanLabels
from utils import cd
from ImportAedat import import_aedat
from ExtractEventsToFramesAndMeanLabels import ExtractEventsToFramesAndMeanLabels
import os
from pathlib import Path, PureWindowsPath
from scipy import io
import numpy as np
import math

# Set the paths of code repository folder, data folder and output folder
# where to generate files of accumulated events.
#F:\MOSAIC Lab\DHP19\Github\lifting_events_to_3d_hpe\scripts\dhp19\generate_DHP19
rootCodeFolder = 'F:/MOSAIC Lab/DHP19/Github/lifting_events_to_3d_hpe/scripts/dhp19/' # root directory of the git repo.
rootDataFolder = 'F:/MOSAIC Lab/DHP19/Dataset/' # root directory of the data downloaded from resiliosync.
outDatasetFolder = 'F:/MOSAIC Lab/DHP19/data/new_vowel/'

###########################################################################

# Cameras number and resolution. Constant for DHP19.
nbcam = 4
sx = 346
sy = 260

########### PARAMETERS: ###################################################

# Average num of events per camera, for constant count frames.
eventsPerFrame = 7500

# ca# Flag and sizes for subsampling the original DVS resolution.
# If no subsample, keep (sx,sy) original img size.
do_subsampling = False #change to False
reshapex = sx #int(sx/2) #change to sx
reshapey = sy #int(sy/2) #change to sy

# Flag to save accumulated recordings.
saveHDF5 = True

# Flag to convert labels
convert_labels = True

save_log_special_events = False
###########################################################################

# Hot pixels threshold (pixels spiking above threshold are filtered out).
thrEventHotPixel = 1*(10**4)

# Background filter: events with less than dt (us) to neighbors pass through.
dt = 70000

### Masks for IR light in the DVS frames.
# Mask 1
xmin_mask1 = 780
xmax_mask1 = 810
ymin_mask1 = 115
ymax_mask1 = 145
# Mask w2
xmin_mask2 = 346*3 + 214
xmax_mask2 = 346*3 + 221
ymin_mask2 = 136
ymax_mask2 = 144

### Paths     #############################################################
t = datetime.now().strftime("%Y_%m_%d_%H%M%S")

DVSrecFolder = os.path.join(rootDataFolder,'DVS_movies/') 
viconFolder = os.path.join(rootDataFolder,'Vicon_data/') 
# output directory where to save files after events accumulation.
out_folder_append = "npy_dataset_" + str(eventsPerFrame) + "_events/"

# Setup output folder path, according to accumulation type and spatial resolution.
outputFolder = os.path.join(outDatasetFolder, out_folder_append,str(reshapex)+"x"+str(reshapey))

log_path = os.path.join(outDatasetFolder, out_folder_append)
log_file = "%s/log_generation_%sx%s_%s.log"%(log_path,reshapex,reshapey, t)
###########################################################################

numConvertedFiles = 0

# setup output folder
if not (os.path.isdir(outputFolder) and os.path.exists(outputFolder)):
    os.mkdir(outputFolder) 

cd(outputFolder)
# log files
with open("%s/Fileslog_%s.log"%(log_path, t), 'w') as fileID:
  if save_log_special_events:
    fileID_specials = open("%s/SpecialEventsLog_%s.log"%(log_path, t), 'w')


  ###########################################################################
  # Loop over the subjects/sessions/movements.

  numSubjects = 17
  numSessions = 5

  fileIdx = 0

  print('Start')
  for subj in range(1,numSubjects+1):
    subj_string = "S%d"%(subj)
    sessionsPath = os.path.join(DVSrecFolder, subj_string)
      
    for sess in range(1,numSessions+1):
      sessString = "session%d"%(sess)
            
      movementsPath = os.path.join(sessionsPath, sessString)
              
      if sess == 1: numMovements = 8
      elif sess == 2: numMovements = 6
      elif sess == 3: numMovements = 6
      elif sess == 4: numMovements = 6
      elif sess == 5: numMovements = 7
              
      for mov in range(1,numMovements+1):
        fileIdx = fileIdx+1
                  
        movString = 'mov%d'%(mov)
                    
        aedatPath = os.path.join(movementsPath, movString +'.aedat')
                    
        # skip iteration if recording is missing.
        if not(os.path.isfile(aedatPath)==True):
          continue

                    
        print(str(fileIdx) +' '+ str(aedatPath))
        recLabel = (subj_string+'_'+str(sess)+'_'+str(mov)+'.mat')

        labelPath = os.path.join(viconFolder, recLabel)
        assert(os.path.isfile(labelPath)==True)
                    
        # name of .h5 output file
        outDVSfile = subj_string+'_'+sessString+'_'+movString+'_'+str(eventsPerFrame)+'events'
                    
        out_file = os.path.join(str(PureWindowsPath(outputFolder)), 'S%d_session_%d_mov_%d_%d_events'%(subj, sess, mov, eventsPerFrame))
                    
                    
        # extract and accumulate if the output file is not already
        # generated and if the input aedat file exists.
        if not(os.path.isfile(out_file+'.h5') == True) and (os.path.isfile(aedatPath)==True):
          aedat = import_aedat({'filePathAndName': PureWindowsPath(movementsPath, movString+'.aedat'), 'fileHandle' : movString+'.aedat', 'dataTypes' :{'polarity', 'special'}})

          XYZPOS = io.loadmat(labelPath)
                          
          events = aedat['data']['polarity']['timeStamp'] 
                          
          ### conditions on special events ###
          try:
            specialEvents = aedat['data']['special']['timeStamp'] 
            numSpecialEvents = len(specialEvents)
                              
            if save_log_special_events:
            # put the specialEvents to string, to print to file.
              specials_=''
              for k in range(0, np.size(specialEvents)):
                  specials_ = specials_ + ' '+ str(specialEvents(k))
              fileID_specials.write('%s \t %s\n'%(aedatPath, specials_))
            # log_special_events
                              
            n = len(XYZPOS['XYZPOS']['head'][0][0])*10000
                              
            if numSpecialEvents == 0:
              # the field aedat.data.special does not exist
              # for S14_5_3. There are no other cases.
              raise Exception('special field is there but is empty')                     
            elif numSpecialEvents == 1:                    
              if (specialEvents-min(events)) > (max(events)-specialEvents):
                # The only event is closer to the end of the recording.
                stopTime = specialEvents
                startTime = (stopTime - n)
              else:
                startTime = specialEvents
                stopTime = startTime + n          
            elif (numSpecialEvents == 2) or (numSpecialEvents == 4):
              # just get the minimum value, the others are max 1
              # timestep far from it.
              special = specialEvents[0] #min(specialEvents)                
              ### special case, for S14_1_1 ###
                # if timeStamp overflows, then get events only
                # until the overflow.
              if events[-1] < events[0]:
                startTime = special
                stopTime = max(events)
                ### regular case ###
              else:
                if (special-events[0]) > (events[-1]-special):
                    # The only event is closer to the end of the recording.
                    stopTime = special
                    startTime = (stopTime - n)
                else:
                    startTime = special
                    stopTime = startTime + n          
            elif (numSpecialEvents == 3) or (numSpecialEvents == 5):
                # in this case we have at least 2 distant special
                # events that we consider as start and stop.
                startTime = specialEvents[0]
                stopTime = specialEvents[-1]
            elif numSpecialEvents > 5:
                # Two recordings with large number of special events.
                # Corrupted recordings, skipped.
                continue              
          except:
            # if no special field exists, get first/last regular
            # events (not tested).
            startTime = events[0]
            stopTime = events[-1]
            
            if save_log_special_events:
              print("** Field 'special' does not exist: " + aedatPath)
              fileID_specials.write('%s \t\n'%(aedatPath))
            # end try reading special events
                          
          print('Processing file: ' + outDVSfile)
          print('Tot num of events in all cameras: ' + str(eventsPerFrame*nbcam))
          # Manually choose the function you want to use to generate constant-count or spatio-temporal frames
          ExtractEventsToVoxelAndMeanLabels(fileID, aedat, events, eventsPerFrame*nbcam, startTime,stopTime,out_file,
          #ExtractEventsToFramesAndMeanLabels(fileID, aedat, events, eventsPerFrame*nbcam, startTime,stopTime,out_file,
              XYZPOS,sx,sy,nbcam,thrEventHotPixel, dt, xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, xmin_mask2, 
              xmax_mask2, ymin_mask2, ymax_mask2, do_subsampling, reshapex, reshapey, saveHDF5,convert_labels)
        else:
          print('%d, File already esists: %s\n', numConvertedFiles, out_file)
          numConvertedFiles = numConvertedFiles +1

  if save_log_special_events:
      fileID_specials.close()

