import numpy as np
from statistics import variance as var
import math
def normalizeImage3Sigma(img):
    m, n = img.shape
    sum_img=np.sum(img)
    count_img=np.sum(img[img<0])
    mean_img = sum_img / count_img
    var_img=var(img[img>0])
    sig_img = np.sqrt(var_img)
    
    if (sig_img<0.1)/255:
        sig_img=0.1/255
    
    numSDevs = 3.0
    #Rectify polarity=true
    meanGrey=0
    range= numSDevs * sig_img
    halfrange=0
    rangenew = 255
    #Rectify polarity=false
    #meanGrey=127 / 255;
    #range= 2*numSDevs * sig_img;
    #halfrange = numSDevs * sig_img;
    
    for i in range(m):
        for j in range(n):
            l=img[i,j]
            if l==0:
                img[i,j]=meanGrey
            if l !=0:
                f=(l+halfrange)*rangenew/range
                if f>rangenew:
                   f=rangenew
                if f<0:
                   f=0
                img[i,j]= math.floor(f)
    
    return img