#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:52:56 2019

@author: bao
"""

import sys
import pandas as pd
sys.path.append('..')
from utils.get_settings import parse
import numpy as np

settings = parse("../utils")




def remove_AB(name_list, A, B):
    names = A.split(' ')+B.split(' ')
    names = [n.lower() for n in names]
    return [i for i in name_list if i[0].lower() not in names]

def get_vector_index(name,offset,token_map):
    name = "".join(name.lower().split(" "))
    idx = 0
    s = ""
    res = []
    for i in sorted(token_map.keys()):
        idx = idx + 1
        if i < offset:
            continue
        else:
            s = s+token_map[i]
            res.append(idx)
            if s == name:
                break
    return np.array(res)

def del_idx(idx_1,idx_2):
    return sorted(set(idx_1)-set(idx_2))



def get_name_vector_index(name_list,token_map):
    idx = []
    for (n,o) in name_list:
        idx.extend(get_vector_index(n,o,token_map))
    return idx



for key,option in settings.items():
    df = pd.read_pickle(option["pickle_path"])
    
    df['name_idx_bert'] = df.apply(lambda x: get_name_vector_index(x["name_list"],x['token_map_bert']), axis=1)
    df['neither_list']  =  df.apply(lambda x:remove_AB(x['name_list'],x["A"],x["B"]),axis = 1)
    df['neither_idx_bert'] = df.apply(lambda x: get_name_vector_index(x["neither_list"],x['token_map_bert']), axis=1)
    
    df['neither_idx_bert'] = df.apply(lambda x: del_idx(x["neither_idx_bert"],x['A_idx_bert']), axis=1)
    df['neither_idx_bert'] = df.apply(lambda x: del_idx(x["neither_idx_bert"],x['B_idx_bert']), axis=1)
    df['neither_idx_bert'] = df.apply(lambda x: del_idx(x["neither_idx_bert"],x['pron_idx_bert']), axis=1)

    df['name_idx_bert'] = df.apply(lambda x: del_idx(x["name_idx_bert"],x['A_idx_bert']), axis=1)
    df['name_idx_bert'] = df.apply(lambda x: del_idx(x["name_idx_bert"],x['B_idx_bert']), axis=1)
    df['name_idx_bert'] = df.apply(lambda x: del_idx(x["name_idx_bert"],x['pron_idx_bert']), axis=1)
    
    df['name_vector_bert_1024'] = df.apply(lambda x: x["vector_bert_1024"][x['name_idx_bert'],:], axis=1)
    df['neither_vector_bert_1024'] = df.apply(lambda x: x["vector_bert_1024"][x['neither_idx_bert'],:], axis=1)
    
    df['name_vector_bert_256'] = df.apply(lambda x: x["vector_bert_256"][x['name_idx_bert'],:], axis=1)
    df['neither_vector_bert_256'] = df.apply(lambda x: x["vector_bert_256"][x['neither_idx_bert'],:], axis=1)

    df.to_pickle(option["pickle_path"])
    if key != 'stage2':
        df = pd.read_pickle(option["pickle_path_augument"])
        
        df['name_idx_bert'] = df.apply(lambda x: get_name_vector_index(x["name_list"],x['token_map_bert']), axis=1)
        df['neither_list']  =  df.apply(lambda x:remove_AB(x['name_list'],x["A"],x["B"]),axis = 1)
        df['neither_idx_bert'] = df.apply(lambda x: get_name_vector_index(x["neither_list"],x['token_map_bert']), axis=1)
    
        df['neither_idx_bert'] = df.apply(lambda x: del_idx(x["neither_idx_bert"],x['A_idx_bert']), axis=1)
        df['neither_idx_bert'] = df.apply(lambda x: del_idx(x["neither_idx_bert"],x['B_idx_bert']), axis=1)
        df['neither_idx_bert'] = df.apply(lambda x: del_idx(x["neither_idx_bert"],x['pron_idx_bert']), axis=1)

        df['name_idx_bert'] = df.apply(lambda x: del_idx(x["name_idx_bert"],x['A_idx_bert']), axis=1)
        df['name_idx_bert'] = df.apply(lambda x: del_idx(x["name_idx_bert"],x['B_idx_bert']), axis=1)
        df['name_idx_bert'] = df.apply(lambda x: del_idx(x["name_idx_bert"],x['pron_idx_bert']), axis=1)
    
        df['name_vector_bert_1024'] = df.apply(lambda x: x["vector_bert_1024"][x['name_idx_bert'],:], axis=1)
        df['neither_vector_bert_1024'] = df.apply(lambda x: x["vector_bert_1024"][x['neither_idx_bert'],:], axis=1)
        
        df['name_vector_bert_256'] = df.apply(lambda x: x["vector_bert_256"][x['name_idx_bert'],:], axis=1)
        df['neither_vector_bert_256'] = df.apply(lambda x: x["vector_bert_256"][x['neither_idx_bert'],:], axis=1)
        
        df.to_pickle(option["pickle_path_augument"])