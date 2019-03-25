# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:03:41 2019

@author: bao
"""
import pandas as pd
import sys
sys.path.append('..')
from utils.get_settings import parse

settings = parse("../utils")

for option in settings.values():
    data = pd.read_csv(option["file_path"],sep = '\t')
    data.to_pickle(option["pickle_path"])


