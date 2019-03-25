#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:10:09 2019

@author: bao
"""

import pandas as pd
import sys
sys.path.append('..')
from utils.get_settings import parse
from feature_function import *

settings = parse("../utils")

for key,option in settings.items():
    df = pd.read_pickle(option["pickle_path"])
    
    print ("Processing token map and sentence map")
    df['token_map_bert'] = df.apply(lambda x: get_token_map(x.Text, x.bert_tokens), axis=1)
    df['sentence_map_bert'] = df.Text.map(get_sentence_map)
    
    print ("Processing dist bert")
    df['A_dist_bert'] = df.apply(lambda x: get_distance(x.Text, x["A-offset"],x['Pronoun-offset']), axis=1)
    df['B_dist_bert'] = df.apply(lambda x: get_distance(x.Text, x["B-offset"],x['Pronoun-offset']), axis=1)

    print ("Processing pos bert")
    df['A_pos_bert'] = df.apply(lambda x: get_relative_pos(x.Text, x["A-offset"],x['sentence_map_bert']), axis=1)
    df['B_pos_bert'] = df.apply(lambda x: get_relative_pos(x.Text, x["B-offset"],x['sentence_map_bert']), axis=1)
    df['pron_pos_bert'] = df.apply(lambda x: get_relative_pos(x.Text, x["Pronoun-offset"],x['sentence_map_bert']), axis=1)

    print ("Processing idx bert")
    df['A_idx_bert'] = df.apply(lambda x: get_vector_index(x.A, x["A-offset"],x['token_map_bert']), axis=1)
    df['B_idx_bert'] = df.apply(lambda x: get_vector_index(x.B, x["B-offset"],x['token_map_bert']), axis=1)
    df['pron_idx_bert'] = df.apply(lambda x: get_vector_index(x.Pronoun, x["Pronoun-offset"],x['token_map_bert']), axis=1)

    print ("Processing vector bert")    
    df['A_vector_bert_1024'] = df.apply(lambda x: x["vector_bert_1024"][x['A_idx_bert'],:], axis=1)
    df['B_vector_bert_1024'] = df.apply(lambda x: x["vector_bert_1024"][x['B_idx_bert'],:], axis=1)
    df['pron_vector_bert_1024'] = df.apply(lambda x: x["vector_bert_1024"][x['pron_idx_bert'],:], axis=1)

    print ("Processing topic")    
    df['topic'] = df['URL'].apply(lambda x: x.split('/')[-1].split('_'))
    df['topic_A'] = df.apply(lambda x: is_topic(x['A'].split(' '), x['topic']), axis=1)
    df['topic_B'] = df.apply(lambda x: is_topic(x['B'].split(' '), x['topic']), axis=1)


    df.to_pickle(option["pickle_path"])
