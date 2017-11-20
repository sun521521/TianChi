# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:04:55 2017

@author: Dorsey
"""

import pandas as pd

WIFI_NOISE_THRE = -85
def split_wifi(user_wifi_info):
    bssid_inten_states = user_wifi_info.split(';')
    split_bssid_inten_states = pd.Series(bssid_inten_states).apply(lambda x: x.split('|'))
    split_bssid_inten_states = list(split_bssid_inten_states)
    split_bssid_inten_states.sort(key = lambda x: int(x[1]), reverse = True)    
    return split_bssid_inten_states

def reduce_noises(line):
    cleaned = []
    bssids = []
    for wifi in line:
        if (not wifi[0] in bssids and int(wifi[1]) > WIFI_NOISE_THRE):
            cleaned.append(wifi)
            bssids.append(wifi[0])
    return cleaned
