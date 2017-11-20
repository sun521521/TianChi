# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:32:21 2017

@author: Dorsey
"""
import gc
import os
import pandas as pd
import datetime

import tools

evaluation_path = os.path.dirname(os.getcwd()) + '\data\evaluation_public.csv'
evaluation = pd.read_csv(evaluation_path)
user_shop_total2 = evaluation.rename(columns = {'longitude':'user_longitude','latitude':'user_latitude'})
del evaluation
gc.collect()

print '数据清洗：对user_shop_total2中的每个user的wifi强度排序.........'
start = datetime.datetime.now()
if not os.path.exists('sorted_user_shop_total2.csv'):
    sorted_user_shop_total2 = user_shop_total2
    sorted_user_shop_total2['wifi_infos'] = user_shop_total2['wifi_infos'].apply(tools.split_wifi)    
    sorted_user_shop_total2.to_csv('sorted_user_shop_total2.csv')
else:
    sorted_user_shop_total2 = pd.read_csv('sorted_user_shop_total2.csv')
    sorted_user_shop_total2['wifi_infos'] = sorted_user_shop_total2['wifi_infos'].apply(lambda x: eval(x))
print datetime.datetime.now() - start 
del user_shop_total2
gc.collect()  

print '数据清洗：删掉user_shop_total2中每个user的wifi噪声.........'
start = datetime.datetime.now()
if not os.path.exists('cleaned_user_shop_total2.csv'):
    cleaned_user_shop_total2 = sorted_user_shop_total2
    cleaned_user_shop_total2['wifi_infos'] = sorted_user_shop_total2['wifi_infos'].apply(tools.reduce_noises)
    cleaned_user_shop_total2.rename(columns={'Unnamed: 0':'un'}, inplace = True)
    cleaned_user_shop_total2.to_csv('cleaned_user_shop_total2.csv')
else:
    cleaned_user_shop_total2 = pd.read_csv('cleaned_user_shop_total2.csv')
    cleaned_user_shop_total2['wifi_infos'] = cleaned_user_shop_total2['wifi_infos'].apply(lambda x: eval(x))
print datetime.datetime.now() - start         

del sorted_user_shop_total2
gc.collect()