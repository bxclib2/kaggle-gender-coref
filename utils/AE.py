#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:29:27 2019

@author: bao
"""

import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self,dim,hidden):
        super(autoencoder, self).__init__()
        self.encoder = nn.Linear(dim, hidden,bias = False)
        self.decoder = nn.Linear(hidden, dim,bias = False)

    def forward(self, x):
        h = self.encoder(x)
        x_ = self.decoder(h)
        return x_,h,x-x_
    
class dimension_reduction(nn.Module):
    def __init__(self,mean1 = None,mean2 = None,mean3 = None,ppa1 = (1024,4),pca = (1024,256),ppa2 = (256,4)):
        super(dimension_reduction, self).__init__()
        if mean1 is not None:
            self.mean1 = nn.Parameter(mean1)
        else:
            self.mean1 = nn.Parameter(torch.zeros(1,ppa1[0]))
        if mean2 is not None:
            self.mean2 = nn.Parameter(mean2)
        else:
            self.mean2 = nn.Parameter(torch.zeros(1,pca[0]))
        if mean3 is not None:
            self.mean3 = nn.Parameter(mean3)
        else:
            self.mean3 = nn.Parameter(torch.zeros(1,ppa2[0]))
        self.ppa1 = autoencoder(*ppa1)
        self.pca = autoencoder(*pca)
        self.ppa2 = autoencoder(*ppa2)
    def forward(self,x):
        _,_,r1 = self.ppa1(x-self.mean1)
        _,h,_ = self.pca(r1-self.mean2)
        _,_,r2 = self.ppa2(h-self.mean3)
        return r2
        