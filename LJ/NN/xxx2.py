# -*- coding: utf-8 -*-
import tensorflow as tf
from bn import *
import numpy as np
from test_pro import *
import pandas as pd

# Input_size = 39
# Min_after_dequeue = 2000
# Batch_Size = 32
# Capacity = Min_after_dequeue + Batch_Size
# epochs = 500
# 
# 
# files = tf.train.match_filenames_once(r"D:\222\xx.csv")
# filename_queue = tf.train.string_input_producer(files,num_epochs=None,shuffle = False)
# 
# reader = tf.TextLineReader()
# key,value = reader.read(filename_queue)
# record_defaults = [[0.] for i in range(40)]#[[1.], [2.], [3.], [4.],[5.]]
# col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40= tf.decode_csv(value,record_defaults=record_defaults)
# x,y = tf.stack([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39]),[col40]
# x = tf.reshape(x,[Input_size])
# y = tf.reshape(y,[1])
# x_input1,y_real1 = tf.train.shuffle_batch([x,y],batch_size=Batch_Size,capacity =Capacity,min_after_dequeue=Min_after_dequeue)
# 
# 
# with tf.Session() as sess:
#     sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
#     print(sess.run(files))
#     corrd = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess = sess, coord = corrd)
#     
#     for i in range(10):
#         print(i)
#         x1,y1 = sess.run([x_input1,y_real1])
#     corrd.request_stop()
#     corrd.join(threads)



valid = np.loadtxt(open("D:\\333\\train10\\10\\train_median_valid.csv","rb"),delimiter=",",skiprows=0)
tt1,tt2,tt3,tt4,tt5,tty = ttt(valid)
res = pd.DataFrame(tty)
maxd = 23.62
mind = 3.07
res = mind + (maxd-mind)*res
res.to_csv("D:\\333\\train10\\10\\NM2.csv",index=None) 


dd1 = pd.read_csv("D:\\333\\train10\\10\\NMtest.csv")
dd2 = pd.read_csv("D:\\333\\train10\\10\\NM2.csv")
# print(dd)
a  = dd1.iloc[:,0]
b  = dd2.iloc[:,0]
print(a.shape,b.shape)
c = np.sum(np.square(a-b))/(2*a.shape[0])
print(c)






