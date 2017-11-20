# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 22:09:27 2017

@author: Dorsey
"""
import datetime
import os
import time
import gc
import pandas as pd
import math
from math import radians, cos, sin, asin, sqrt    
from itertools import chain

import handle_imbalance
import perform_train_test2

#参数
INTEN_LAMDA = 0.02
#BSSID_LAMDA = 0.15
DISTANCE_LAMDA = 0.013
TIME_THRESHOLD = '2017-08-31 23:50:00'
FEATURE_NAMES = ['inten_hist', 'bssid_hist','distance_hist']
#FEATURE_NAMES = ['inten_hist', 'bssid_hist', 'distance_hist']

#计算经纬度距离
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）  
    # 将十进制度数转化为弧度  
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    # haversine公式  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 # 地球平均半径，单位为公里  
    return c * r * 1000  

print '构建特征：加载cleaned_user_shop_total.........'
start = datetime.datetime.now()
if not 'cleaned_user_shop_total' in dir():
    cleaned_user_shop_total_path =  os.path.dirname(os.getcwd()) + '\DataCleaning\cleaned_user_shop_total.csv'
    cleaned_user_shop_total = pd.read_csv(cleaned_user_shop_total_path)
    cleaned_user_shop_total['wifi_infos'] = cleaned_user_shop_total['wifi_infos'].apply(lambda x: eval(x))
print datetime.datetime.now() - start


print '构建特征：加载cleaned_user_shop_total2.........'
start = datetime.datetime.now()
if not 'cleaned_user_shop_total2' in dir():
    cleaned_user_shop_total_path2 =  os.path.dirname(os.getcwd()) + '\DataCleaning\cleaned_user_shop_total2.csv'
    cleaned_user_shop_total2 = pd.read_csv(cleaned_user_shop_total_path2)
    cleaned_user_shop_total2['wifi_infos'] = cleaned_user_shop_total2['wifi_infos'].apply(lambda x: eval(x))
print datetime.datetime.now() - start


print '构建特征：加载mall_to_wifis.........'
start = datetime.datetime.now()
if not 'mall_to_wifis' in dir():
    mall_to_wifis_path =  os.path.dirname(os.getcwd()) + '\preprocessing\mall_to_wifis.csv'
    mall_to_wifis = pd.read_csv(mall_to_wifis_path)
    mall_to_wifis['wifis_sum'] = mall_to_wifis['wifis_sum'].apply(lambda x: eval(x))
    mall_to_wifis = mall_to_wifis.groupby(['mall_id']).sum()
print datetime.datetime.now() - start

print '构建特征：加载discriminative_wifis_to_shop.........'
start = datetime.datetime.now()
discriminative_wifis_to_shop_path =  os.path.dirname(os.getcwd()) + '\preprocessing\discriminative_wifis_to_shop.csv'
discriminative_wifis_to_shop = pd.read_csv(discriminative_wifis_to_shop_path)
discriminative_wifis_to_shop['wifis_mean_inten'] = discriminative_wifis_to_shop['wifis_mean_inten'].apply(lambda x: eval(x))
print datetime.datetime.now() - start


print '构建特征：把每个shop对应的固定wifi的bssid：inten字典放在大字典中'
start = datetime.datetime.now()
shop_wifi = discriminative_wifis_to_shop['wifis_mean_inten']
shop_wifi.index = discriminative_wifis_to_shop['shop_id']
shop_wifi = shop_wifi.to_dict()
print datetime.datetime.now() - start

def get_inten_hist(line, shops_belong, fixed_wifis_dict, record):
    ##--------------提取inten_hist特征--------------
    #inten特征初始化
    inten_hist = {}
    inten_hist = {'inten_' + shop:0 for shop in shops_belong}

    #找到该样本最大的强度
    if len(fixed_wifis_dict) == 0:
        print 'fixed_wifis_dict为空:', record
        line['features'] = {}
        return line
    max_wifi_inten = max(fixed_wifis_dict.values())
    
    #计算inten_hist
    increment_nums_dict = {k:0 for k in inten_hist.keys()}
    
    for wifi in fixed_wifis_dict:
        p_to_accept = 1-(1-math.exp(-INTEN_LAMDA*abs(fixed_wifis_dict[wifi] - max_wifi_inten)))
        
        for bin in inten_hist:
            mall_shop_wifis = shop_wifi[ bin[6:] ]
         
            if wifi in mall_shop_wifis:
                increment = p_to_accept* (1-(1-math.exp(-INTEN_LAMDA*abs(fixed_wifis_dict[wifi] - mall_shop_wifis[wifi])))) 
                inten_hist[bin] = inten_hist[bin] + increment
                increment_nums_dict[bin] = increment_nums_dict[bin] + 1 
    hist1 = {k:inten_hist[k]/increment_nums_dict[k] for k in inten_hist.keys() if increment_nums_dict[k] != 0 }
    hist2 = {k:0 for k in inten_hist.keys() if increment_nums_dict[k] == 0 }
    inten_hist = dict(hist2, **hist1)
    #对inten_hist归一化    
    max_inten = max(inten_hist.values())
    
#    if max_inten != 0:
    inten_hist = {k:v/(1.0*max_inten) for k,v in inten_hist.items()}
            
    line['features'] = dict(line['features'], **inten_hist)
    return line

def get_bssid_hist(line, shops_belong, fixed_wifis_dict):
    ##--------------提取bssid_hist特征--------------
    #bssid特征初始化
    bssid_hist = {}
    bssid_hist = {'bssid_' + shop:0 for shop in shops_belong}
    
    if len(fixed_wifis_dict) == 0:
#        line['features'] = dict(line['features'], **bssid_hist)
        line['features'] = {}
        return line

    for bin in bssid_hist:       
        mall_shop_wifis = shop_wifi[ bin[6:] ]
        bssid_hist[bin] = bssid_hist[bin] + len([wifi for wifi in fixed_wifis_dict if wifi in mall_shop_wifis])
    
#    #找bssid_hist中最大数
#    max_num = max(bssid_hist.values())
    
    #计算bssid_hist
    hist1 = {k:1.0*bssid_hist[k]/len(shop_wifi[ k[6:] ]) for k in bssid_hist.keys() if len(shop_wifi[ k[6:] ]) != 0}
    hist2 = {k:0 for k in bssid_hist.keys() if len(shop_wifi[ k[6:] ]) == 0}
    
    
#    hist1 = {k:1-(1-math.exp(-BSSID_LAMDA*abs(bssid_hist[k] - max_num))) for k in bssid_hist.keys() if bssid_hist[k] != 0}
#    hist2 = {k:0 for k in bssid_hist.keys() if bssid_hist[k] == 0}
    bssid_hist = dict(hist2, **hist1)

    #bssid_hist归一化    
    max_value = max(bssid_hist.values())    
    if max_value != 0:
        bssid_hist = {k:v/(1.0*max_value) for k,v in bssid_hist.items()}
 
    
    line['features'] = dict(line['features'], **bssid_hist)
    return line

def get_distance_hist(line, shops_belong, record):
    ##--------------提取distance_hist特征--------------    
    #distance特征初始化
    distance_hist = {}
    distance_hist = {'distance_' + shop:0 for shop in shops_belong}    
    
    #计算经纬度距离
    distance_hist = {k:haversine(lon[ k[9:] ], lat[ k[9:] ], record['user_longitude'], record['user_latitude']) for k in distance_hist.keys()}

    #计算distance_hist    
    distance_hist = {k:1-(1-math.exp(-DISTANCE_LAMDA*distance_hist[k])) for k in distance_hist.keys()}
    
    #distance_hist归一化    
    max_value = max(distance_hist.values())    
    if max_value != 0:
        distance_hist = {k:v/(1.0*max_value) for k,v in distance_hist.items()}
    
    line['features'] = dict(line['features'], **distance_hist)
    return line

def get_time_hist(line, record):
    ##--------------提取time_hist特征--------------    
    #time特征初始化
    time_hist = {}
    time_point = range(0, 24, 3)
    time_hist = {'time_' + str(i):0 for i in time_point}    
    
    #计算time_hist 
    time_stamp = str(record['time_stamp'])
    hour = time.strptime(time_stamp, '%Y-%m-%d %H:%M').tm_hour
    time_hist[ 'time_' + str(time_point[hour/3]) ] = 1

    line['features'] = dict(line['features'], **time_hist)
    return line


print '构建特征：计算mall:shops字典.........'
start = datetime.datetime.now()
new_discriminative_wifis_to_shop = discriminative_wifis_to_shop
new_discriminative_wifis_to_shop['shop_id'] = new_discriminative_wifis_to_shop['shop_id'].apply(lambda x: [x])
mall_to_shops = new_discriminative_wifis_to_shop.loc[:,['mall_id','shop_id']].groupby('mall_id').sum()

mall_shop_dict = mall_to_shops['shop_id']
mall_shop_dict.index = mall_to_shops.index
mall_shop_dict = mall_shop_dict.to_dict()
print datetime.datetime.now() - start

#global g,h 
#g = list() 
#h = list()
def generate_features(record):
    line = record
    shops_belong = mall_shop_dict[ record['mall_id'] ]
   
    #在样本的wifi列表中，筛选出该mall的固定wifi
    mall_wifis_set = mall_to_wifis.at[record['mall_id'], 'wifis_sum']
    fixed_wifis_dict = {}
    
    for wifi in record['wifi_infos']:
        if wifi[0] in mall_wifis_set:
            fixed_wifis_dict[wifi[0]]  = int(wifi[1])
    
#    if record['shop_id'] == 's_504218':
#        g.append([fixed_wifis_dict])
#        h.append(record['wifi_infos'])

    line['features'] = {}
    for name in FEATURE_NAMES:
        if name == 'inten_hist':
            line = get_inten_hist(line, shops_belong, fixed_wifis_dict, record)
        if name == 'bssid_hist':
            line = get_bssid_hist(line, shops_belong, fixed_wifis_dict)
        if name == 'distance_hist':
            line = get_distance_hist(line, shops_belong, record)
        if name == 'time_hist':
            line = get_time_hist(line, record)
        
    line['features'] = [line['features']]
    
    global user_num
    user_num = user_num + 1
    print user_num
    return line

user_num = 0
shop_infos_path = os.path.dirname(os.getcwd()) + '\data\ccf_first_round_shop_info.csv'
shop_infos = pd.read_csv(shop_infos_path)

#把经纬度放在字典里
longitude = shop_infos['longitude']
longitude.index = shop_infos['shop_id']
lon = longitude.to_dict()

latitude = shop_infos['latitude']
latitude.index = shop_infos['shop_id']
lat = latitude.to_dict()

def generate_labels(fetures_shop):
    label = fetures_shop['label']
    nums = len(fetures_shop['features'])
    fetures_shop['label'] = [label] * nums
    return fetures_shop

mall_names_set = set(mall_to_shops.index)
length = len(mall_names_set)
mall_num = 0

log = pd.DataFrame()
#mall_names_set = ['m_1263']
for mall in mall_names_set:
    
    mall_num += 1
    print '构建特征：对于%s(%d/%d),生成多分类器所需要的特征.........'%(mall, mall_num, length)
    start = datetime.datetime.now()
    mall_records = cleaned_user_shop_total[cleaned_user_shop_total['mall_id'] == mall]
    mall_features = mall_records.apply(generate_features, axis = 1)
    mall_features = mall_features.loc[:,['time_stamp','shop_id','features']]
    
    mall_records2 = cleaned_user_shop_total2[cleaned_user_shop_total2['mall_id'] == mall]
    mall_features2 = mall_records2.apply(generate_features, axis = 1)
    mall_features2 = mall_features2.loc[:,['row_id','features']]
    print datetime.datetime.now() - start

    print '模型训练：对于%s(%d/%d),划分训练集和验证集.........'%(mall, mall_num, length)
    start = datetime.datetime.now()    
    mall_features_train = mall_features[mall_features['time_stamp'] < TIME_THRESHOLD]
    mall_features_train = mall_features_train.loc[:,['shop_id','features']]
   
    mall_features_test = mall_features[mall_features['time_stamp'] >= TIME_THRESHOLD]
    mall_features_test = mall_features_test.loc[:,['shop_id','features']]
    print datetime.datetime.now() - start
    
    print '模型训练：对于%s(%d/%d),处理训练集的样本不平衡问题.........'%(mall, mall_num, length)
    start = datetime.datetime.now()    
    train_test = handle_imbalance.reduce_imbalance(mall_features_train, mall_features_test, mall_shop_dict, mall)
    train_set = train_test[0]
    train_set['label'] = train_set.index
    test_set = train_test[1]
    test_set['label'] = test_set.index
    
    train_set_labels = train_set.apply(generate_labels, axis = 1)
    test_set_labels = test_set.apply(generate_labels, axis = 1)
    print datetime.datetime.now() - start
    
    print '模型训练：对于%s(%d/%d),得到训练集和验证集的特征矩阵和标签.........'%(mall, mall_num, length)
    start = datetime.datetime.now()    
    #训练集
    features_list = [x for x in train_set_labels['features']]
    train_features = list(chain(*features_list))
    labels_list = [x for x in train_set_labels['label']]
    train_labels = list(chain(*labels_list))
    
    #验证集
    features_list = [x for x in test_set_labels['features']]
    test_features = list(chain(*features_list))
    labels_list = [x for x in test_set_labels['label']]
    test_labels = list(chain(*labels_list))
   
    #评测集
    features_list = [x for x in mall_features2['features']]
    online_test_features = list(chain(*features_list))
    row_id = list(mall_features2['row_id'])
    print datetime.datetime.now() - start

    gc.collect()
    #训练和验证
    log = perform_train_test2.train_test2(log, mall, mall_num, length, train_features, train_labels, test_features, test_labels,online_test_features, row_id)
    
    