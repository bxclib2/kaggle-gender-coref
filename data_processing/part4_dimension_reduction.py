#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:58:50 2019

@author: bao
"""

import sys
import pandas as pd
import torch
sys.path.append('..')
from utils.get_settings import parse
from utils.AE import *

settings = parse("../utils")

d_r = dimension_reduction()
d_r.load_state_dict(torch.load("../dimension_reduction/DR_1024.pth"))

for key,option in settings.items():
    print ("Processing "+ key)
    df = pd.read_pickle(option["pickle_path"])
    df['vector_bert_256'] = df.vector_bert_1024.apply(lambda x: d_r(torch.Tensor(x)).data.numpy())
    
    print ("Processing vector bert")    
    df['A_vector_bert_256'] = df.apply(lambda x: x["vector_bert_256"][x['A_idx_bert'],:], axis=1)
    df['B_vector_bert_256'] = df.apply(lambda x: x["vector_bert_256"][x['B_idx_bert'],:], axis=1)
    df['pron_vector_bert_256'] = df.apply(lambda x: x["vector_bert_256"][x['pron_idx_bert'],:], axis=1)
    df.to_pickle(option["pickle_path"])