import math
import numpy as np

def subsample(I,sizex,sizey,new_size_x,new_size_y, where_to_cut):
    block_w=math.floor(sizex/new_size_x)
    block_h=math.floor(sizey/new_size_y)

    #this cuts always initial part
    if where_to_cut == 'begin':
        I_cut=I[(sizex-new_size_x*block_w):,(sizey-new_size_y*block_h):]
    
    # this cuts always final part
    if where_to_cut== 'end':
        I_cut=I[:-(sizex-new_size_x*block_w),:-(sizey-new_size_y*block_h)]
    
    # better to cut the central part. Cut central part for cameras 0,2,3.
    # camera 1 has different subsample function
    if where_to_cut== 'center':
        I_cut=I[int((sizex-new_size_x*block_w)/2):-1-int((sizex-new_size_x*block_w)/2),int((sizey-new_size_y*block_h)/2):-1-int((sizey-new_size_y*block_h)/2)]

    
    Isub=np.zeros((new_size_x,new_size_y))
    for w in range(new_size_x):
        for h in range(0, new_size_y):
            x_start=w*block_w
            y_start=h*block_h
            Isub[w,h]=np.mean(I_cut[x_start:x_start+block_w,y_start:y_start+block_h])
        
    
    return Isub
