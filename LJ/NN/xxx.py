# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd

# a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# b = a[:,0:1]
# c = a[:,2:3]
# d = np.concatenate((b,c),axis = 1)
# 
# print(b,c,d.shape)

# my_matrix = np.loadtxt(open("D:\\333\\train_median_valid3.csv","rb"),delimiter=",",skiprows=0)  
# print(my_matrix.shape)

a =pd.read_csv(r'D:\333\NNtest.csv',header = None)
b = a.round({0:3})
b.to_csv(r'C:\Users\LiJing\Desktop\NNtest3.csv')