# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 17:12:02 2017

@author: Dorsey
"""

'''
利用SVM进行分类
'''


import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer 
import datetime
from sklearn.svm import SVC    
import os
import gc

def svm_find_para(log, arr, y, feature_names,arr_to_test, y_test, mall):    
    model = SVC()    
    model.decision_function_shape = 'ovo'
    
    c_sets = [1, 10,30,50,70]
    g_sets = [0.1,0.3,0.5,0.9]
    kernel_sets = ['rbf']
    
    best_C = 0
    best_g = 0
    best_ker = 'linear'
    best_acc = 0
    for ker in kernel_sets:
        if ker == 'linear':
            g_sets = [0.005]
        for c in c_sets:
            for g in g_sets:
                start = datetime.datetime.now() 
                model = SVC(kernel = ker, C = c, gamma = g)    
                model.fit(arr, y)
                y_test_predict = model.predict(arr_to_test)
                                      
                results = [y_test_predict[k] == y_test[k] for k in range(0, len(y_test))]
                acc = 1.0*sum(results)/len(y_test)
                new_log = pd.DataFrame([{'mall':mall, 'kernel':ker, 'c':c, 'gamma':g, 'acc':acc}])
                print new_log
                print '用时:', datetime.datetime.now() - start
      
                if acc > best_acc:                    
                    best_C = c
                    best_g = g
                    best_ker = ker
                    best_acc = acc
                    y_test_predict_best = y_test_predict
                    
                log = pd.concat([log, new_log])
                log.to_csv('log.csv')
                gc.collect()
    print '----mall:',mall,'best_C:',best_C,'best_g:',best_g,'best_acc:',best_acc           
    model = SVC(C = best_C, gamma = best_g, kernel = best_ker)    
    model.decision_function_shape = 'ovo'

    return model, y_test_predict_best, log, best_acc


def train_test(log, mall, mall_num, length, train_features, train_labels, test_features, test_labels,online_test_features, row_id):
    vec = DictVectorizer()
    arr = vec.fit_transform(train_features).toarray()
    arr_to_test = vec.fit_transform(test_features).toarray()
    arr_to_online_test = vec.fit_transform(online_test_features).toarray()    
    
    
    y = np.array(train_labels)
    y_test = np.array(test_labels)    
    
    feature_names = vec.get_feature_names()
    
    print '模型训练：对于%s(%d/%d),进行模型训练与验证'%(mall, mall_num, length)    
    model_predict_log = svm_find_para(log, arr, y, feature_names, arr_to_test, y_test, mall)
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


    model.fit(arr, y)
    y_onlie_test_predict = model.predict(arr_to_online_test)
    

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    