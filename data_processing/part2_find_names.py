#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:55:49 2019

@author: bao
"""

import nltk
from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')
import sys
import pandas as pd
sys.path.append('..')
from utils.get_settings import parse

settings = parse("../utils")

def spans(txt):
    span = []
    tokens=nltk.word_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        span.append((token, offset))
    return tokens, span

def extract_name(text):
    tokens, span = spans(text)
    name_span = []
    tags = st.tag(tokens)
    for ind, tag in enumerate(tags):
        if tag[1]=='PERSON': 
            name_span.append(span[ind])
    return name_span

for key,option in settings.items():
    print ("Processing "+ key)
    df = pd.read_pickle(option["pickle_path"])
    df['name_list'] = df.Text.apply(lambda x: extract_name(x))
    df.to_pickle(option["pickle_path"])