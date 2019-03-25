# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:03:41 2019

@author: bao
"""


try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0


def parse(UTIL_DIR):
    # instantiate
    config = ConfigParser()
    
    # parse existing file
    config.read(UTIL_DIR+'/config.ini')

    
    settings = {}
    
    process = config.get("default","process")
    
    options = ["train","test","valid","stage2"]
    for option in options:
        if option in process:
            settings[option] = {}
            settings[option]["file_path"] = config.get("default",option+"_file_path")
            settings[option]["pickle_path"] = config.get("default",option+"_pickle")
            if option != "stage2":
                settings[option]["pickle_path_augument"] = config.get("default",option+"_pickle_a")
                

    return settings    
    
