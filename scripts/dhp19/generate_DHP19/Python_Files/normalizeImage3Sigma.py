### File created by preshma

import numpy as np
from statistics import variance as var
import math
def normalizeImage3Sigma(img):
    m, n = img.shape
    sum_img=np.sum(img)
    count_img=np.count_nonzero(img>0)
    mean_img = sum_img / count_img
    var_img=var(img[img>0])
    sig_img = np.sqrt(var_img)
    
    if (sig_img<0.1)/255:
        sig_img=0.1/255
    
    numSDevs = 3.0
    meanGrey=0
    range_= numSDevs * sig_img
    halfrange=0
    rangenew = 255
    
    for i in range(m):
        for j in range(n):
            if img[i,j]==0:
                img[i,j]=meanGrey
            else:
                f=(img[i,j]+halfrange)*rangenew/range_
                if f>rangenew: f=rangenew
                if f<0: f=0
                img[i,j]= math.floor(f)
    
    return img