# -*- coding: utf-8 -*-
import tensorflow as tf
from bn import *
import numpy as np
from test_pro import *
import pandas as pd

Input_size = 39
Min_after_dequeue = 5000
Batch_Size = 32
Capacity = Min_after_dequeue + Batch_Size
epochs = 1000


#files = tf.train.match_filenames_once(r"D:\333\train10\10\train_median_train.csv")
files = tf.train.match_filenames_once(r"D:\333\train_median_all.csv")
filename_queue = tf.train.string_input_producer(files,num_epochs=None,shuffle = False)

reader = tf.TextLineReader()
key,value = reader.read(filename_queue)
record_defaults = [[0.] for i in range(40)]#[[1.], [2.], [3.], [4.],[5.]]
col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40= tf.decode_csv(value,record_defaults=record_defaults)
x,y = tf.stack([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39]),[col40]
x = tf.reshape(x,[Input_size])
y = tf.reshape(y,[1])
x_input1,y_real1 = tf.train.shuffle_batch([x,y],batch_size=Batch_Size,capacity =Capacity,min_after_dequeue=Min_after_dequeue)


#input size
input_size = [10,6,5,7,19]
h1_nums = [1,1,1,1,1]
output_size = 1
lr = 0.01
regularation_rate = 0.002
regularizer = tf.contrib.layers.l2_regularizer(regularation_rate)
BN_enable = True
is_train = True

def get_weight(size,name,regularizer):
    weights = tf.get_variable(name,size,initializer=tf.truncated_normal_initializer(stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection("losses",regularizer(weights))
    return weights

def get_bias(size,name):
    bias = tf.get_variable(name,size,initializer=tf.constant_initializer(0.0))
    return bias


input_1 = tf.placeholder(tf.float32, [None,input_size[0]], name = "input_1")
input_2 = tf.placeholder(tf.float32, [None,input_size[1]], name = "input_2")
input_3 = tf.placeholder(tf.float32, [None,input_size[2]], name = "input_3")
input_4 = tf.placeholder(tf.float32, [None,input_size[3]], name = "input_4")
input_5 = tf.placeholder(tf.float32, [None,input_size[4]], name = "input_5")
y_real = tf.placeholder(tf.float32, [None,1], name = "y_real")
        
#------------------------------------------------------------------------------------    
    
with tf.variable_scope("h1_1"):
    weights = get_weight([input_size[0],h1_nums[0]],"weight_h1_1", regularizer=regularizer)
    bias = get_bias([h1_nums[0]], "bias_h1_1")
    accumulate = tf.matmul(input_1,weights)+bias
    if BN_enable:
        bn_1_1 = batch_norm(name = "bn_h1_1")
        accumulate = bn_1_1(accumulate,train = is_train)
    layer1_1 = tf.nn.relu(accumulate)
with tf.variable_scope("h1_2"):
    weights = get_weight([input_size[1],h1_nums[1]],"weight_h1_2", regularizer=regularizer)
    bias = get_bias([h1_nums[1]], "bias_h1_2")
    accumulate = tf.matmul(input_2,weights)+bias
    if BN_enable:
        bn_1_2 = batch_norm(name = "bn_h1_2")
        accumulate = bn_1_2(accumulate,train = is_train)
    layer1_2 = tf.nn.relu(accumulate)    
with tf.variable_scope("h1_3"):
    weights = get_weight([input_size[2],h1_nums[2]],"weight_h1_3", regularizer=regularizer)
    bias = get_bias([h1_nums[2]], "bias_h1_3")
    accumulate = tf.matmul(input_3,weights)+bias
    if BN_enable:
        bn_1_3 = batch_norm(name = "bn_h1_3")
        accumulate = bn_1_3(accumulate,train = is_train)
    layer1_3 = tf.nn.relu(accumulate)     
with tf.variable_scope("h1_4"):
    weights = get_weight([input_size[3],h1_nums[3]],"weight_h1_4", regularizer=regularizer)
    bias = get_bias([h1_nums[3]], "bias_h1_4")
    accumulate = tf.matmul(input_4,weights)+bias
    if BN_enable:
        bn_1_4 = batch_norm(name = "bn_h1_4")
        accumulate = bn_1_4(accumulate,train = is_train)
    layer1_4 = tf.nn.relu(accumulate) 
with tf.variable_scope("h1_5"):
    weights = get_weight([input_size[4],h1_nums[4]],"weight_h1_5", regularizer=regularizer)
    bias = get_bias([h1_nums[4]], "bias_h1_5")
    accumulate = tf.matmul(input_5,weights)+bias
    if BN_enable:
        bn_1_5 = batch_norm(name = "bn_h1_5")
        accumulate = bn_1_5(accumulate,train = is_train)
    layer1_5 = tf.nn.relu(accumulate) 
    
#----------------------------------------------------------------------------------------
    
with tf.variable_scope("h2_1"):
    weights = get_weight([h1_nums[0],1],"weight_h2_1", regularizer=regularizer)
    bias = get_bias([1], "bias_h2_1")
    accumulate = tf.matmul(layer1_1,weights)+bias
    if BN_enable:
        bn_2_1 = batch_norm(name = "bn_h2_1")
        accumulate = bn_2_1(accumulate,train = is_train)
    layer2_1 = tf.nn.relu(accumulate)
with tf.variable_scope("h2_2"):
    weights = get_weight([h1_nums[1],1],"weight_h2_2", regularizer=regularizer)
    bias = get_bias([1], "bias_h2_2")
    accumulate = tf.matmul(layer1_2,weights)+bias
    if BN_enable:
        bn_2_2 = batch_norm(name = "bn_h2_2")
        accumulate = bn_2_2(accumulate,train = is_train)
    layer2_2 = tf.nn.relu(accumulate)
with tf.variable_scope("h2_3"):
    weights = get_weight([h1_nums[2],1],"weight_h2_3", regularizer=regularizer)
    bias = get_bias([1], "bias_h2_3")
    accumulate = tf.matmul(layer1_3,weights)+bias
    if BN_enable:
        bn_2_3 = batch_norm(name = "bn_h2_3")
        accumulate = bn_2_3(accumulate,train = is_train)
    layer2_3 = tf.nn.relu(accumulate)
with tf.variable_scope("h2_4"):
    weights = get_weight([h1_nums[3],1],"weight_h2_4", regularizer=regularizer)
    bias = get_bias([1], "bias_h2_4")
    accumulate = tf.matmul(layer1_4,weights)+bias
    if BN_enable:
        bn_2_4 = batch_norm(name = "bn_h2_4")
        accumulate = bn_2_4(accumulate,train = is_train)
    layer2_4 = tf.nn.relu(accumulate)    
with tf.variable_scope("h2_5"):
    weights = get_weight([h1_nums[4],1],"weight_h2_5", regularizer=regularizer)
    bias = get_bias([1], "bias_h2_5")
    accumulate = tf.matmul(layer1_5,weights)+bias
    if BN_enable:
        bn_2_5 = batch_norm(name = "bn_h2_5")
        accumulate = bn_2_5(accumulate,train = is_train)
    layer2_5 = tf.nn.relu(accumulate)
    
#----------------------------------------------------------------------------------------

with tf.variable_scope("output"):
    weights = get_weight([5,1],"output_weights", regularizer=regularizer)
    bias = get_bias([1], "output_bias")
    output_input = c = tf.concat([layer1_1,layer1_2,layer1_3,layer1_4,layer1_5], axis=1)
    y_predict = tf.nn.sigmoid(tf.matmul(output_input,weights)+bias)
    
#----------------------------------------------------------------------------------------    
with tf.name_scope("cal_loss"):    
    loss_entropy = tf.reduce_mean(tf.square(tf.subtract(y_predict, y_real)))
    final_loss = loss_entropy *(23.62-3.07)*(23.62-3.07)+ tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(final_loss)
#     tf.summary.scalar('losssssss', final_loss)
# summary_op =  tf.summary.merge_all()
saver = tf.train.Saver()


with tf.Session() as sess:
#     saver.restore(sess, "D:\\111\model_1\model_2.ckpt")
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    print(sess.run(files))
    corrd = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = corrd)
                  
    test = np.loadtxt(open("D:\\333\\train10\\10\\train_median_valid.csv","rb"),delimiter=",",skiprows=0) 
    t1,t2,t3,t4,t5,ty = ttt(test)
    ty = ty.reshape((ty.shape[0],1))
                 
    for i in range(epochs):
        x1,y1 = sess.run([x_input1,y_real1])
        i1,i2,i3,i4,i5 = ttt2(x1)
        _,accuary = sess.run([train_step,final_loss],feed_dict = {input_1:i1,input_2:i2,input_3:i3,input_4:i4,input_5:i5,y_real:y1})
        print(accuary)
        if i%20 == 0:
            accuary1 = sess.run([final_loss],feed_dict = {input_1:t1,input_2:t2,input_3:t3,input_4:t4,input_5:t5,y_real:ty})
            print("test-------------------------:%s" % accuary1)
#         summary_writer = tf.summary.FileWriter("D:\\111\\",sess.graph)
#         summary_writer.add_summary(op, i)
#             
    saver.save(sess,"D:\\111\model_8\model_xxx.ckpt")
#     corrd.request_stop()
#     corrd.join(threads)
        
#     valid = np.loadtxt(open("D:\\333\\test_median.csv","rb"),delimiter=",",skiprows=0)
#     tt1,tt2,tt3,tt4,tt5 = ttt2(valid)
# #     tt1,tt2,tt3,tt4,tt5,tty = ttt(valid)
# #     tty = tty.reshape((tty.shape[0],1))
#     is_train = False
#     predict = sess.run([y_predict],feed_dict = {input_1:tt1,input_2:tt2,input_3:tt3,input_4:tt4,input_5:tt5})
#     res = pd.DataFrame(predict[0])
#     maxd = 23.62
#     mind = 3.07
#     res = mind + (maxd-mind)*res
#     res.to_csv("D:\\333\\NNtest.csv",index=None)  
#      
