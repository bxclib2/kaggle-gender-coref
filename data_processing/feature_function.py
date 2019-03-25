#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:19:07 2019

@author: bao
"""
import spacy
import numpy as np
spacy_model = "en_core_web_sm"
nlp = spacy.load(spacy_model)


def get_token_map(sentence,token_list):
    token_map = {}
    i = 0
    #print (token_list)
    token_list = token_list[1:-1]
    #print (token_list)
    for t in token_list:
        #print (i)
        if t!= "#":
            t = t.strip("#")
        while sentence[i:i+len(t)].lower()!=t:
            #print (sentence[i:i+len(t)].lower())
            i = i + 1
        token_map[i] = t
        #print (token_map)
        i = i + len(t)
    return token_map

def get_sentence_map(sentence):
    doc = nlp(sentence)
    sentence_map = {}
    i = 0

    for s in doc.sents:
        s = str(s)
        while sentence[i:i+len(s)]!=s:
            i = i + 1
        sentence_map[i] = s
        i = i + len(s)
    return sentence_map



def get_distance(sentence,A,B):
    start = min(A,B)
    end = max(A,B)
    dist = nlp(sentence[start:end])
    return (B-A)/abs(B-A)*len(dist)/500

def get_relative_pos(sentence,offset,sentence_map):
    for i in sorted(sentence_map.keys()):
        if offset >= i:
            break
    return len(nlp(sentence[i:offset]))/len(sentence_map[i])

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

def is_topic(name, topic):
    for i in name:
        if i in topic:
            return 1.0
    return 0.0