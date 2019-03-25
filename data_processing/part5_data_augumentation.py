#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:00:11 2019

@author: bao
"""

import sys
import pandas as pd
sys.path.append('..')
from utils.get_settings import parse
from feature_function import *
from random import randint

settings = parse("../utils")

def get_neither_name(A, B, name_list):
    AB_list = A.split(' ') + B.split(' ')
    for i in range(len(name_list)):
        if name_list[i][0] in AB_list and i > 0 and i < len(name_list) - 1:
            if name_list[i][1] + len(name_list[i][0]) + 1 == name_list[i+1][1]:
                AB_list.append(name_list[i+1][0])
            elif name_list[i][1] - len(name_list[i][0]) - 1 == name_list[i-1][1]:
                AB_list.append(name_list[i-1][0])
    return [i for i in name_list if i[0] not in AB_list]

def append_name(df):
    if df['A-coref'] == True:
        df['neither_name'].append((df['A'],df['A-offset']))
    if df['B-coref'] == True:
        df['neither_name'].append((df['B'],df['B-offset']))
    return df['neither_name']

def pop_name(df):
    k = randint(0,df["len"]-1)
    if df['A-coref'] == True:
        df['neither_name'].pop(k)
    if df['B-coref'] == True:
        df['neither_name'].pop(k)
    return df['neither_name']

for key,option in settings.items():
    if key != 'stage2':
        print ("Processing "+ key)
        print ("Argumenting")
        df = pd.read_pickle(option["pickle_path"])
        df['neither_name'] = df.apply(lambda x: get_neither_name(x['A'],x['B'],x['name_list']), axis = 1)
        df['len'] = df['neither_name'].apply(lambda x: len(x))
        df_A = df[df['A-coref'] == True]
        df_A = df_A[df_A['len']>0]
    
        df_B = df[df['B-coref'] == True]
        df_B = df_B[df_B['len']>0]
        
        df_A['neither_name'] =  df_A.apply(lambda x: append_name(x), axis=1)
        df_B['neither_name'] =  df_B.apply(lambda x: append_name(x), axis=1)
        
        df_A['A'] = df_A.apply(lambda x: x['neither_name'][0][0], axis=1)
        df_B['B'] = df_B.apply(lambda x: x['neither_name'][0][0], axis=1)
    
        df_A['A-offset'] = df_A.apply(lambda x: x['neither_name'][0][1], axis=1)
        df_B['B-offset'] = df_B.apply(lambda x: x['neither_name'][0][1], axis=1)
    
        df_A['neither_name'] =  df_A.apply(lambda x: pop_name(x), axis=1)
        df_B['neither_name'] =  df_B.apply(lambda x: pop_name(x), axis=1)
    
        df_A['A-coref'] = False
        df_B['B-coref'] = False
    
        df_new = pd.concat([df_A, df_B]) 
        
        df_new = df_new.drop(columns = ['len', 'neither_name'])
        
        df_new.reset_index(inplace=True,drop=True)
        
        print ("Reprocessing dist bert")
        df_new['A_dist_bert'] = df_new.apply(lambda x: get_distance(x.Text, x["A-offset"],x['Pronoun-offset']), axis=1)
        df_new['B_dist_bert'] = df_new.apply(lambda x: get_distance(x.Text, x["B-offset"],x['Pronoun-offset']), axis=1)

        print ("Reprocessing pos bert")
        df_new['A_pos_bert'] = df_new.apply(lambda x: get_relative_pos(x.Text, x["A-offset"],x['sentence_map_bert']), axis=1)
        df_new['B_pos_bert'] = df_new.apply(lambda x: get_relative_pos(x.Text, x["B-offset"],x['sentence_map_bert']), axis=1)
    
        print ("Reprocessing idx bert")
        df_new['A_idx_bert'] = df_new.apply(lambda x: get_vector_index(x.A, x["A-offset"],x['token_map_bert']), axis=1)
        df_new['B_idx_bert'] = df_new.apply(lambda x: get_vector_index(x.B, x["B-offset"],x['token_map_bert']), axis=1)
  

        print ("Reprocessing vector bert")    
        df_new['A_vector_bert_1024'] = df_new.apply(lambda x: x["vector_bert_1024"][x['A_idx_bert'],:], axis=1)
        df_new['B_vector_bert_1024'] = df_new.apply(lambda x: x["vector_bert_1024"][x['B_idx_bert'],:], axis=1)
        df_new['pron_vector_bert_1024'] = df_new.apply(lambda x: x["vector_bert_1024"][x['pron_idx_bert'],:], axis=1)
        
        print ("Rerocessing vector bert")    
        df_new['A_vector_bert_256'] = df_new.apply(lambda x: x["vector_bert_256"][x['A_idx_bert'],:], axis=1)
        df_new['B_vector_bert_256'] = df_new.apply(lambda x: x["vector_bert_256"][x['B_idx_bert'],:], axis=1)
        df_new['pron_vector_bert_256'] = df_new.apply(lambda x: x["vector_bert_256"][x['pron_idx_bert'],:], axis=1)
        
        df_new.to_pickle(option["pickle_path_augument"])