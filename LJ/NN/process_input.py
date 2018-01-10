# -*- coding: utf-8 -*- 
import pandas as pd
import numpy as np


df_train = pd.read_csv(r'D:\d_test_A_20180102.csv',encoding='GB2312')
for c, dtype in zip(df_train.columns, df_train.dtypes):
    if dtype == np.float64:
        df_train[c] = df_train[c].astype(np.float32)
# df_train = df_train[df_train['血糖'] < 30]
print(len(df_train))
# print(max(df_train['血糖']),min(df_train['血糖']))
# # #  
df_train.fillna(df_train.median(),inplace = True)
# # df_train.fillna(0,inplace = True)
df_train = df_train.replace('男',1)
df_train = df_train.replace('女',0)
df_train = df_train.replace('??',0)
ttrain  = df_train.drop(['id','体检日期'], axis=1)
ttrain.to_csv(r'D:\333\test_median.csv',index=False,encoding='GB2312')
#       
# df_train = pd.read_csv(r'D:\333\train_median.csv',encoding='GB2312')
# print(df_train.shape)
# df_train2 = df_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# df_train2.to_csv(r'D:\333\train_median_one.csv',index = False,encoding='GB2312')
#   
# # 
 
# df_train = pd.read_csv(r'D:\333\train_median_one.csv',encoding='GB2312')
# print(df_train.shape)
# df2 = df_train.sample(frac=1) .reset_index(drop=True)
# print(df2[:564].shape)
# df2[:564].to_csv(r'D:\333\train10\10\train_median_valid.csv',index = False,encoding='GB2312')
# df2[564:].to_csv(r'D:\333\train10\10\train_median_train.csv',index = False,encoding='GB2312')    


# df = pd.read_csv(r'D:\Eclipse\Projects\Tianchi\data\train\after\train_median_train.csv',header = None,encoding='UTF-8')
# df.to_csv(r'D:\Eclipse\Projects\Tianchi\data\train\after\train_median_train2.csv',index=False,encoding='utf-8')