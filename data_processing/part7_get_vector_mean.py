#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:21:50 2019

@author: bao
"""

import sys
import pandas as pd
sys.path.append('..')
from utils.get_settings import parse
import numpy as np

settings = parse("../utils")

def label(A,B):
    if A is True:
        return 0
    if B is True:
        return 1
    return 2

for key,option in settings.items():
    df = pd.read_pickle(option["pickle_path"])
    
    df["A_vector_bert_1024_mean"] = df["A_vector_bert_1024"].map(lambda x:np.mean(x,axis = 0))
    df["B_vector_bert_1024_mean"] = df["B_vector_bert_1024"].map(lambda x:np.mean(x,axis = 0))
    df["pron_vector_bert_1024_mean"] = df["pron_vector_bert_1024"].map(lambda x:np.mean(x,axis = 0))
    df["product_vector_A_bert_1024"] = df["A_vector_bert_1024_mean"]*df["pron_vector_bert_1024_mean"]
    df["product_vector_B_bert_1024"] = df["B_vector_bert_1024_mean"]*df["pron_vector_bert_1024_mean"]
    
    
    df["A_vector_bert_256_mean"] = df["A_vector_bert_256"].map(lambda x:np.mean(x,axis = 0))
    df["B_vector_bert_256_mean"] = df["B_vector_bert_256"].map(lambda x:np.mean(x,axis = 0))
    df["pron_vector_bert_256_mean"] = df["pron_vector_bert_256"].map(lambda x:np.mean(x,axis = 0))
    df["product_vector_A_bert_256"] = df["A_vector_bert_256_mean"]*df["pron_vector_bert_256_mean"]
    df["product_vector_B_bert_256"] = df["B_vector_bert_256_mean"]*df["pron_vector_bert_256_mean"]
    if key != "stage2":
        df["label"] = df.apply(lambda x:label(x["A-coref"],x["B-coref"]),axis = 1)


    df.to_pickle(option["pickle_path"])
    if key != 'stage2':
        df = pd.read_pickle(option["pickle_path_augument"])
        
        df["A_vector_bert_1024_mean"] = df["A_vector_bert_1024"].map(lambda x:np.mean(x,axis = 0))
        df["B_vector_bert_1024_mean"] = df["B_vector_bert_1024"].map(lambda x:np.mean(x,axis = 0))
        df["pron_vector_bert_1024_mean"] = df["pron_vector_bert_1024"].map(lambda x:np.mean(x,axis = 0))
        df["product_vector_A_bert_1024"] = df["A_vector_bert_1024_mean"]*df["pron_vector_bert_1024_mean"]
        df["product_vector_B_bert_1024"] = df["B_vector_bert_1024_mean"]*df["pron_vector_bert_1024_mean"]
        df["label"] = df.apply(lambda x:label(x["A-coref"],x["B-coref"]),axis = 1)
        
        
        df["A_vector_bert_256_mean"] = df["A_vector_bert_256"].map(lambda x:np.mean(x,axis = 0))
        df["B_vector_bert_256_mean"] = df["B_vector_bert_256"].map(lambda x:np.mean(x,axis = 0))
        df["pron_vector_bert_256_mean"] = df["pron_vector_bert_256"].map(lambda x:np.mean(x,axis = 0))
        df["product_vector_A_bert_256"] = df["A_vector_bert_256_mean"]*df["pron_vector_bert_256_mean"]
        df["product_vector_B_bert_256"] = df["B_vector_bert_256_mean"]*df["pron_vector_bert_256_mean"]
        
        df.to_pickle(option["pickle_path_augument"])