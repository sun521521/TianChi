# -*- coding: utf-8 -*-

"""
Created on Tue Oct 24 22:28:33 2017

@author: Dorsey
"""

import os
import pandas as pd
import re
import datetime
import gc

import tools

user_shop_path = os.path.dirname(os.getcwd()) + '\data\ccf_first_round_user_shop_behavior.csv'
user_shop = pd.read_csv(user_shop_path)

#展示wifi强度分布
print '数据清洗：计算wifi强度分布.........'
start = datetime.datetime.now()
shops_wifis = user_shop.loc[:, ['shop_id','wifi_infos']]
wifi_infos_new = shops_wifis['wifi_infos'].apply(lambda x: x + ';')
shops_wifis_new = pd.concat([pd.DataFrame(shops_wifis['shop_id']), pd.DataFrame(wifi_infos_new)], axis = 1)
shop_to_wifis = shops_wifis_new.groupby('shop_id').sum()
intens = shop_to_wifis['wifi_infos'].apply(lambda x: re.findall('-\d+', x))

def counts(line):
    line = pd.Series(line)
    return line.value_counts()

intens_hist = intens.apply(counts).sum() # 不同强度出现的次数
intens_hist.plot() # 1. 重要信号； 2. 过度信号；3. 隔壁信号；4.无用信号（清洗）
print datetime.datetime.now() - start 

#连接ccf_first_round_shop_info.csv和ccf_first_round_user_shop_behavior.csv
shop_infos_path = os.path.dirname(os.getcwd()) + '\data\ccf_first_round_shop_info.csv'
shop_infos = pd.read_csv(shop_infos_path)

shop_infos = shop_infos.rename(columns = {'longitude':'shop_longitude','latitude':'shop_latitude'})
user_shop = user_shop.rename(columns = {'longitude':'user_longitude','latitude':'user_latitude'})
user_shop_total = pd.merge(shop_infos, user_shop, on = 'shop_id')


print '数据清洗：对每个user的wifi强度排序.........'
start = datetime.datetime.now()
if not os.path.exists('sorted_user_shop_total.csv'):
    sorted_user_shop_total = user_shop_total
    sorted_user_shop_total['wifi_infos'] = user_shop_total['wifi_infos'].apply(tools.split_wifi)    
    sorted_user_shop_total.to_csv('sorted_user_shop_total.csv')
else:
    sorted_user_shop_total = pd.read_csv('sorted_user_shop_total.csv')
    sorted_user_shop_total['wifi_infos'] = sorted_user_shop_total['wifi_infos'].apply(lambda x: eval(x))
print datetime.datetime.now() - start 
#有格式的DataFrame等变量一般不要保存为CSV，需要用pickle.jump()
#若保存为了CSV，则需要把误认为是字符串的地方重新编码

del user_shop_total
del intens  
del intens_hist  
del shop_infos   
del shop_to_wifis
del shops_wifis
del shops_wifis_new    
del user_shop
del wifi_infos_new
gc.collect()


print '数据清洗：删掉每个user的wifi噪声.........'
start = datetime.datetime.now()
if not os.path.exists('cleaned_user_shop_total.csv'):
    cleaned_user_shop_total = sorted_user_shop_total
    cleaned_user_shop_total['wifi_infos'] = sorted_user_shop_total['wifi_infos'].apply(tools.reduce_noises)
    cleaned_user_shop_total.rename(columns={'Unnamed: 0':'row_id'}, inplace = True)
    cleaned_user_shop_total.to_csv('cleaned_user_shop_total.csv')
else:
    cleaned_user_shop_total = pd.read_csv('cleaned_user_shop_total.csv')
    cleaned_user_shop_total['wifi_infos'] = cleaned_user_shop_total['wifi_infos'].apply(lambda x: eval(x))
print datetime.datetime.now() - start         

del sorted_user_shop_total 
gc.collect()    



