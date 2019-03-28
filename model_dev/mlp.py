#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:29:00 2019

@author: bao
"""

import torch
import torch.nn as nn
import numpy as np

np.random.seed(seed=0)
torch.manual_seed(0)    

class MLP_(nn.Module):
    def __init__(self,dim,hidden,dropout_rate):
        super(MLP_, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, 3)
        )
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[-1].weight)
    def forward(self, x):
        x = self.layers(x)
        return x
    

class MLP():
    def __init__(self,dim,hidden = 37,dropout_rate = 0.9,weight = [1.0,1.0,1.0],l2 = (0.03,0.09)):
        self.mlp = MLP_(dim,hidden,dropout_rate).cuda()
        self.EPOCHS = 30
        self.batch_size = 25
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.Tensor(weight)).cuda()
        self.l2 = l2
    def fit(self,X_train,y_train):
        for e in range(self.EPOCHS):
            for b in range(0,X_train.shape[0],self.batch_size):
                self.mlp.train()
                start = b
                end = min(X_train.shape[0],b+self.batch_size)
                batch_data = X_train[start:end,:]
                batch_label = y_train[start:end]
                
                output = self.mlp(torch.Tensor(batch_data).cuda())
                batch_label = torch.LongTensor(batch_label).cuda()
                loss = self.loss_fn(output,batch_label)

                l2_norm = torch.norm(self.mlp.layers[0].weight, p=2)
                loss += l2_norm*self.l2[0]
                
                l2_norm = torch.norm(self.mlp.layers[-1].weight, p=2)
                loss += l2_norm*self.l2[1]
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        print ("[MLP]",end = "")
        return self
    def predict_proba(self,X_test):
        self.mlp.eval()
        pred_mlp = []
        with torch.no_grad():
            for b in range(0,X_test.shape[0],self.batch_size):
                start = b
                end = min(X_test.shape[0],b+self.batch_size)
                batch_data = X_test[start:end,:]
                batch_data = torch.Tensor(batch_data)
                pred_mlp_batch = torch.nn.Softmax(dim = 1)(self.mlp(batch_data.cuda())).cpu().data.numpy()
                pred_mlp.append(pred_mlp_batch.copy())
        return np.concatenate(pred_mlp,axis = 0)
                
        
    

    
