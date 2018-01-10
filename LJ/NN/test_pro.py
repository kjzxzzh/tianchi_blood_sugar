# -*- coding: utf-8 -*-
import numpy as np

def ttt(x):
    x1,y = x[:,:-1],x[:,-1]
    base = x1[:,0:2]
    p1 = x1[:,2:10]
    p2 = x1[:,10:14]
    p3 = x1[:,14:17]
    p4 = x1[:,17:22]
    p5 = x1[:,22:39]
    i1 = np.concatenate((base,p1),axis = 1)
    i2 = np.concatenate((base,p2),axis = 1)
    i3 = np.concatenate((base,p3),axis = 1)
    i4 = np.concatenate((base,p4),axis = 1)
    i5 = np.concatenate((base,p5),axis = 1)
    return i1,i2,i3,i4,i5,y

def ttt2(x):
    x1  = x
#     y = y.reshape((y.shape[0],1))
#     print(y.shape)
    base = x1[:,0:2]
    p1 = x1[:,2:10]
    p2 = x1[:,10:14]
    p3 = x1[:,14:17]
    p4 = x1[:,17:22]
    p5 = x1[:,22:39]
    i1 = np.concatenate((base,p1),axis = 1)
    i2 = np.concatenate((base,p2),axis = 1)
    i3 = np.concatenate((base,p3),axis = 1)
    i4 = np.concatenate((base,p4),axis = 1)
    i5 = np.concatenate((base,p5),axis = 1)
    return i1,i2,i3,i4,i5