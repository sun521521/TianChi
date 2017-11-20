# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 14:09:16 2017

@author: Dorsey
"""
import pandas as pd
import numpy as np
import random
SMALL_SAMPLE_RATIO = 0.4
RADIUS = 0.08

def reduce_imbalance(mall_features_train, mall_features_test, mall_shop_dict, mall):
    
    mall_shops_set = set(mall_shop_dict[mall])  
   
#     该mall的训练集中，每个shop的平均人流量
    mean_users_num = len(mall_features_train)/len(mall_shops_set)
    mean_users_num_test = len(mall_features_test)/len(mall_shops_set)    
    
    mall_features_train = mall_features_train.groupby('shop_id').sum()
    mall_features_test = mall_features_test.groupby('shop_id').sum()    
    
    #补齐训练集，防止部分shop为空
    shops_train_set = set(mall_features_train.index)
    train_lack = mall_shops_set - shops_train_set
    train_lack_features = mall_features_test.loc[list(train_lack), ['features']]
    mall_features_train = pd.concat([mall_features_train, train_lack_features])  
    
    #补齐验证集
    shops_test_set = set(mall_features_test.index)
    test_lack = mall_shops_set - shops_test_set
    test_lack_features = mall_features_train.loc[list(test_lack), ['features']]
    mall_features_test = pd.concat([mall_features_test, test_lack_features])  
    
    #对训练集增加人工样本
    def add_records(shop_features):
        features_dict_list = shop_features['features']
        amount = len(features_dict_list)
        
        for i in [17, 11, 5, 3,1]:
            if amount > i*mean_users_num:
                shop_features['features'] = features_dict_list[0:len(features_dict_list):i]
                return shop_features

        if (amount <= mean_users_num and amount > SMALL_SAMPLE_RATIO*mean_users_num):
            return shop_features
        
        nums_to_add = SMALL_SAMPLE_RATIO*mean_users_num - amount
        mean_dict = pd.DataFrame(features_dict_list).apply(lambda x: np.mean(x)).to_dict()
        
        dict_to_add = []
        for i in range(0, int(nums_to_add)):
            per_dict_to_add = {k: random.uniform(max(mean_dict[k]-RADIUS,0), min(mean_dict[k]+RADIUS,1)) for k in mean_dict.keys()}
            dict_to_add.append(per_dict_to_add)

        features_dict_list.extend(dict_to_add)
        shop_features['features'] = features_dict_list
        return shop_features
    
    new_mall_features_train = mall_features_train.apply(add_records, axis = 1)
    mean_users_num = mean_users_num_test
    new_mall_features_test = mall_features_test.apply(add_records, axis = 1)
    
    return new_mall_features_train, new_mall_features_test









    