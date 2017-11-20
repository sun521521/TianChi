# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:35:14 2017

@author: Dorsey
"""

import os
import pandas as pd
import datetime
import gc

#参数
SHOPS_NUM_TO_PUBLIC_I = 5
INTEN_TO_PUBLIC_I = -70
SHOPS_NUM_TO_PUBLIC_II = 10
INTEN_TO_PUBLIC_II = -60

print '预处理：加载cleaned_user_shop_total.........'
start = datetime.datetime.now()
if not 'cleaned_user_shop_total' in dir():
    cleaned_user_shop_total_path =  os.path.dirname(os.getcwd()) + '\DataCleaning\cleaned_user_shop_total.csv'
    cleaned_user_shop_total = pd.read_csv(cleaned_user_shop_total_path)
    cleaned_user_shop_total['wifi_infos'] = cleaned_user_shop_total['wifi_infos'].apply(lambda x: eval(x))
print datetime.datetime.now() - start

#预处理第一阶段
print '预处理：找出每个shop所能捕获的wifi，作为该shop的描述'
mall_shop_wifi = cleaned_user_shop_total.loc[:, ['mall_id','shop_id','wifi_infos']]
if not os.path.exists('users_to_shop_count.csv'):
    users_to_shop_count = mall_shop_wifi.groupby(['mall_id','shop_id']).count()    
    users_to_shop_count.to_csv('users_to_shop_count.csv')
else:
    users_to_shop_count = pd.read_csv('users_to_shop_count.csv')

print '预处理：位于每个shop的所有user的wifi汇总.........'
start = datetime.datetime.now()
if not os.path.exists('wifi_sum.csv'):
    wifi_sum = mall_shop_wifi.groupby(['mall_id','shop_id']).sum()
    wifi_sum.to_csv('wifi_sum.csv')
else:
    wifi_sum = pd.read_csv('wifi_sum.csv')
    wifi_sum['wifi_infos'] = wifi_sum['wifi_infos'].apply(lambda x: eval(x))
print datetime.datetime.now() - start 
del mall_shop_wifi

users_to_shop_count = users_to_shop_count.rename(columns = {'wifi_infos':'users_count'})
wifi_sum = wifi_sum.rename(columns = {'wifi_infos':'wifis_sum'})
users_count_wifi = pd.concat([users_to_shop_count, pd.DataFrame(wifi_sum['wifis_sum'])], axis = 1)
#del users_to_shop_count
del wifi_sum
gc.collect()

def delete_mobile_hotspot(record):
    thre_to_hotspot = int(record['users_count']*0.5)
    dict_to_count = {}
    wifis_state = record['wifis_sum']
    dict_to_mean_inten = {}
    
    intens_sum = {}
    for wifi in pd.Series(wifis_state):
        if wifi[0] in dict_to_count:
            dict_to_count[wifi[0]] +=1
            intens_sum[wifi[0]] += int(wifi[1])
        else:
            dict_to_count[wifi[0]] = 1
            intens_sum[wifi[0]] = int(wifi[1]) 
    
    fixed_wifis = []
    for key in dict_to_count:
        if dict_to_count[key] >= thre_to_hotspot or dict_to_count[key] >= 20:
            dict_to_mean_inten[key] = round(intens_sum[key]/dict_to_count[key]) 
            fixed_wifis.append(key)
    record['wifis_sum'] = fixed_wifis # 去除了移动热点，格式为list
    record['wifis_mean_inten'] = dict_to_mean_inten # 对应的wifi平均强度
    return record
            
print '预处理：去除移动热点.........'
start = datetime.datetime.now()

def recovery(x):
    x['wifis_sum'] = eval(x['wifis_sum'])
    x['wifis_mean_inten'] = eval(x['wifis_mean_inten'])
    return x

if not os.path.exists('users_count_wifi.csv'):
    users_count_wifi = users_count_wifi.apply(delete_mobile_hotspot, axis = 1)            
    del users_count_wifi['users_count']
    users_count_wifi.to_csv('users_count_wifi.csv')
    users_count_wifi = pd.read_csv('users_count_wifi.csv')
    users_count_wifi = users_count_wifi.apply(recovery, axis = 1)
else:
    users_count_wifi = pd.read_csv('users_count_wifi.csv')
    users_count_wifi = users_count_wifi.apply(recovery, axis = 1)
print datetime.datetime.now() - start 

#预处理第二阶段
start = datetime.datetime.now()
print '1.按mall划分，求每个shop所能捕获的wifi的并集'
mall_to_wifis = users_count_wifi.loc[:, ['mall_id','wifis_sum']].groupby('mall_id').sum()
mall_to_wifis['wifis_sum'] = mall_to_wifis['wifis_sum'].apply(lambda x: set(x))

print '2.去掉每个mall的公共wifi/没有辨识度的wifi'
def meet_first_rule(wifi, mall_name):
    global users_count_wifi
    mall_to_dict = users_count_wifi[users_count_wifi['mall_id'] == mall_name]
    shops_num = 0
    for dict in mall_to_dict['wifis_mean_inten']:
        if wifi in dict:
            if dict[wifi] >= INTEN_TO_PUBLIC_I:
                return False
            else:
                shops_num += 1
        else:
            continue
    
    if shops_num > SHOPS_NUM_TO_PUBLIC_I:
        return True
    else:
        return False

def meet_second_rule(wifi, mall_name):
    global users_count_wifi
    mall_to_dict = users_count_wifi[users_count_wifi['mall_id'] == mall_name]
    shops_num = 0
    for dict in mall_to_dict['wifis_mean_inten']:
        if wifi in dict:
            if dict[wifi] >= INTEN_TO_PUBLIC_II:
                return False
            else:
                shops_num += 1
        else:
            continue
    
    if shops_num > SHOPS_NUM_TO_PUBLIC_II:
        return True
    else:
        return False

def delete_public_wifis(line):
    wifis = line['wifis_sum']
    mall_name = line['mall_id']
    owner_wifis = set()
    for wifi in wifis:
        if meet_first_rule(wifi, mall_name) or meet_second_rule(wifi, mall_name):
            continue
        else:
            owner_wifis.add(wifi)
    line['wifis_sum'] = owner_wifis
    return line

mall_to_wifis['mall_id'] = mall_to_wifis.index
mall_to_wifis = mall_to_wifis.apply(delete_public_wifis, axis = 1) # 每个shop具有辨识度的wifi集合，格式为set
mall_to_wifis.to_csv('mall_to_wifis.csv')

print '3.得到每个shop具有辨识度的wifi'
def get_discriminative_wifis_to_shop(mall_shop_wifis):
    mall_id = mall_shop_wifis['mall_id']
    wifis = mall_shop_wifis['wifis_mean_inten']
    
    mall_wifis_set = mall_to_wifis.at[mall_id, 'wifis_sum']
    discriminative_wifis = {}
    for wifi in wifis:
        if wifi in mall_wifis_set:
            discriminative_wifis[wifi] = wifis[wifi]
            
    mall_shop_wifis['wifis_mean_inten'] = discriminative_wifis
    return mall_shop_wifis

discriminative_wifis_to_shop = users_count_wifi.apply(get_discriminative_wifis_to_shop, axis = 1)
del users_count_wifi
discriminative_wifis_to_shop  = discriminative_wifis_to_shop.loc[:,['mall_id','shop_id','wifis_mean_inten']]
discriminative_wifis_to_shop.to_csv('discriminative_wifis_to_shop.csv')
print '预处理：去除每个shop的公共wifi/没有辨识度的wifi.........(1.2.3.)总共：', datetime.datetime.now() - start 
gc.collect()

