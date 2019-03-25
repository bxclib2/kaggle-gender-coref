# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:43:35 2019

@author: bao
"""

import pandas as pd
from tqdm import tqdm
from bert_serving.client import BertClient
import sys
sys.path.append('..')
from utils.get_settings import parse


settings = parse("../utils")

def crop_vector(vector,tokens):
    return vector[0:len(tokens),:]
    
    
def get_bert_vector(df):
    bc = BertClient()
    data_num = len(list(df.Text.values)) 
    batch_size = 20
    text = list(df.Text.values)
    text_all = []
    tokens_all = []
    for i in tqdm(range(0,data_num,batch_size),ncols = 100):
        encoded = bc.encode(text[i:min(batch_size+i,data_num)],show_tokens=True)
        encoded_text = list(encoded[0])
        tokens = encoded[1]
        text_all.extend(encoded_text)
        tokens_all.extend(tokens)
    return text_all,tokens_all


for key,option in settings.items():
    df = pd.read_pickle(option["pickle_path"])
    tqdm.write("Processing "+key)
    df["vector_bert_1024"] , df["bert_tokens"] = get_bert_vector(df)
    df["vector_bert_1024"] = df.apply(lambda x: crop_vector(x["vector_bert_1024"], x["bert_tokens"]), axis=1)
    df.to_pickle(option["pickle_path"])
    
    
