# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 17:12:02 2017

@author: Dorsey
"""
'''
利用随机森林进行分类
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer 
import datetime
import os
import gc
from sklearn.ensemble import RandomForestClassifier 

def svm_find_para(log, arr, y, mall, arr_to_test, y_test):       
    n_estimators_sets = [70,80,90,100,110,120,130]
    oob_score_sets = [True] 
    min_weight_fraction_leaf_sets = [0,0.05, 0.1,]
    
    best_estimators = 0
    best_oob = True
    best_fraction = 0
    best_acc = 0
    for estimators in n_estimators_sets:
        for oob in oob_score_sets:
            for fraction in min_weight_fraction_leaf_sets:
                start = datetime.datetime.now() 
                model = RandomForestClassifier(n_estimators = estimators, oob_score = oob, min_weight_fraction_leaf = fraction)    
                model.fit(arr, y)
                y_test_predict = model.predict(arr_to_test)
                                      
                acc = model.oob_score_
                new_log = pd.DataFrame([{'mall':mall, 'estimators':estimators, 'weight':fraction, 'acc':acc}])
                print new_log
                print '用时:', datetime.datetime.now() - start
      
                if acc > best_acc:                    
                    best_estimators = estimators
                    best_fraction = fraction
                    best_acc = acc
                    
                log = pd.concat([log, new_log])
                log.to_csv('log.csv')
                gc.collect()
    print '----mall:',mall,'estimators:',best_estimators,'best_fraction:',best_fraction,'best_acc:',best_acc           
    model = RandomForestClassifier(n_estimators = best_estimators, min_weight_fraction_leaf = best_fraction, oob_score = best_oob)    

    return model, y_test_predict,log, best_acc

def train_test2(log, mall, mall_num, length, train_features, train_labels, test_features, test_labels,online_test_features, row_id):
    vec = DictVectorizer()
    arr = vec.fit_transform(train_features).toarray()
    arr_to_test = vec.fit_transform(test_features).toarray()
    arr_to_online_test = vec.fit_transform(online_test_features).toarray()    
    
    y = np.array(train_labels)
    y_test = np.array(test_labels)    
    
#    feature_names = vec.get_feature_names()
    
    print '模型训练：对于%s(%d/%d),进行模型训练与验证'%(mall, mall_num, length)    
    model_predict_log = svm_find_para(log, arr, y, mall,arr_to_test,y_test)

    model = model_predict_log[0]
    y_test_predict = model_predict_log[1]
    log = model_predict_log[2]
    acc = model_predict_log[3]
    
    results = [y_test_predict[k] == y_test[k] for k in range(0, len(y_test))]
    
#保存每个mall和每个shop的准确率    
    if os.path.exists('acc_mall.csv'):
        total_acc = pd.read_csv('acc_mall.csv')
        if mall in list(total_acc['mall']):
            total_acc = {k:v for k in total_acc['mall'] for v in total_acc['acc']}
            total_acc[mall] = acc
            total_acc = pd.DataFrame(pd.Series(total_acc)).reset_index()
            total_acc.columns = ['mall','acc']
        else:
            per_acc = pd.DataFrame(pd.Series({mall:acc})).reset_index()
            per_acc.columns = ['mall', 'acc']
            total_acc = pd.concat([total_acc, per_acc])
        total_acc.to_csv('acc_mall.csv', index = None)
    else:
        per_acc = pd.DataFrame(pd.Series({mall:acc})).reset_index()
        per_acc.columns = ['mall', 'acc']
        per_acc.to_csv('acc_mall.csv', index = None)

#计算并保存每个商铺的准确率，有利于分析问题
    mall_shop = [ mall + '_'+ y_test[k] for k in range(0, len(y_test)) ]   
    shop_results = pd.Series(results)
    shop_results.index = mall_shop
    shop_results_df = pd.DataFrame(shop_results).reset_index()
    shop_results_df.columns = ['mall_shop', 'results']
    acc = shop_results_df.groupby('mall_shop').mean()
    shop_acc = acc['results']
    shop_acc.index = acc.index
    if os.path.exists('acc_shop.csv'):
        total_acc = pd.read_csv('acc_shop.csv')
        mall_shops = list(total_acc['mall_shop'])
        
        if mall_shop[3] in mall_shops:
            total_acc.index = total_acc['mall_shop']
            total_acc.loc[shop_acc.index, 'acc'] = list(shop_acc)
            total_acc.to_csv('acc_shop.csv', index = None)
        else:
            per_acc_df = pd.DataFrame(shop_acc).reset_index()
            per_acc_df.columns = ['mall_shop', 'acc']
            total_acc = pd.concat([total_acc, per_acc_df])
            total_acc.to_csv('acc_shop.csv', index = None)
    else:
        per_acc = pd.DataFrame(shop_acc).reset_index()
        per_acc.columns = ['mall_shop', 'acc']
        per_acc.to_csv('acc_shop.csv', index = None)

#对评测集进行训练，分类
    model.fit(arr, y)
    y_onlie_test_predict = model.predict(arr_to_online_test)
    
#对最终结果进行保存
    if os.path.exists('result.csv'):
        total_result = pd.read_csv('result.csv')   
        
        if row_id[1] in list(total_result['row_id']):
            total_result.index = total_result['row_id']
            total_result.loc[row_id, 'shop_id'] = list(y_onlie_test_predict)
            total_result.to_csv('result.csv', index = None)
        else:
            per_result = pd.Series(y_onlie_test_predict)
            per_result.index = np.int64(row_id)
            per_result = pd.DataFrame(per_result).reset_index()
            per_result.columns = ['row_id', 'shop_id']
            total_result = pd.concat([total_result, per_result])
            total_result.to_csv('result.csv', index = None)
    else:
        per_result = pd.Series(y_onlie_test_predict)
        per_result.index = np.int64(row_id)
        per_result = pd.DataFrame(per_result).reset_index()
        per_result.columns = ['row_id', 'shop_id']
        per_result.to_csv('result.csv', index = None)
    return log
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    