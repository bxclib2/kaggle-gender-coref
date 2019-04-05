#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:51:11 2019

@author: bao
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
    
vector = "vector_bert_1024"
pron_vector = "pron_vector_bert_1024_mean"

A_idx = "A_idx_bert"
B_idx = "B_idx_bert"
pron_idx = "pron_idx_bert"
neither_idx = "name_idx_bert"

np.random.seed(seed=0)
torch.manual_seed(0)

def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)
    return probs

class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob,fc_size):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))
        self.drop1 = nn.Dropout(self.drop_prob)  # (bs, c_len, hid_size)
        self.drop2 = nn.Dropout(self.drop_prob)
        self.output_layer1 = nn.Linear(4*hidden_size,fc_size)
        self.drop3 = nn.Dropout(self.drop_prob)
        self.output_layer2 = nn.Linear(fc_size,1)
        self.drop4 = nn.Dropout(self.drop_prob)
        nn.init.xavier_uniform_(self.output_layer1.weight)
        nn.init.xavier_uniform_(self.output_layer2.weight)
        
        
    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        
        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        
        x = self.drop3(x)
        
        x = self.output_layer1(x)
        
        x = torch.nn.ELU()(x)
        
        x = self.drop4(x)
        
        x = self.output_layer2(x)
        
        return x.squeeze(-1)

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = self.drop1(c)  # (bs, c_len, hid_size)
        q = self.drop2(q)  # (bs, q_len, hid_size)
        #print (c.size())
        #print (q.size())
        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

    

class BIDAF():
    def __init__(self,hidden_size = 1024 , drop_prob = 0.70, fc_size = 64, weight = [1.0,1.0]):
        self.bidaf = BiDAFAttention(hidden_size = 1024 , drop_prob = 0.70, fc_size = 64).cuda()
        self.EPOCHS = 50
        self.batch_size = 25
        self.opt = torch.optim.Adam(self.bidaf.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.Tensor(weight)).cuda()
    def fit(self,X_train,y_train):
        self.bidaf.train()
        for e in range(self.EPOCHS):
            for b in range(0,X_train.shape[0],self.batch_size):
                start = b
                end = min(X_train.shape[0],b+self.batch_size)

                batch_data = X_train[vector][start:end]
                batch_label = X_train.label[start:end]
                batch_pron = X_train[pron_vector][start:end]
                batch_pron = torch.Tensor(np.array(list(batch_pron.values))).unsqueeze(1).cuda()
                batch_data_torch = pad_sequence([torch.Tensor(v) for v in batch_data]).cuda().transpose(0,1)
                batch_padding = batch_data_torch.mean(dim=1,keepdim = True)#torch.zeros(batch_data.size()[0],1,batch_data.size()[2]).cuda()*0.001
                batch_data_torch = torch.cat([batch_padding,batch_data_torch],dim = 1)
                batch_label = torch.LongTensor(list(batch_label)).cuda()
                c_mask = torch.zeros_like(batch_data_torch.mean(-1,keepdim = True)) != batch_data_torch.mean(-1,keepdim = True)
                q_mask = torch.zeros_like(batch_pron.mean(-1,keepdim = True)) != batch_pron.mean(-1,keepdim = True)
                c_mask = c_mask.cuda()
                q_mask = q_mask.cuda()
                output = self.bidaf(batch_data_torch,batch_pron,c_mask,q_mask)
                mask_A = [np.array(v)+1 for v in list(X_train[A_idx][start:end])]
                mask_B = [np.array(v)+1 for v in list(X_train[B_idx][start:end])]
                mask_neither = [np.array(v)+1 for v in list(X_train[neither_idx][start:end])]
                prob_list = []
                for i,(v_A,v_B,v_neither) in enumerate(zip(mask_A,mask_B,mask_neither)):
                    v_A = torch.LongTensor(v_A).cuda()
                    A_prob_ = output[i,v_A].mean()
                    v_B = torch.LongTensor(v_B).cuda()
                    B_prob_ = output[i,v_B].mean()
                    v_neither = list(v_neither)
                    v_neither.append(0)
                    v_neither = torch.LongTensor(v_neither).cuda()
                    other_prob = output[i,v_neither].mean()
                    prob_list.append(torch.cat([A_prob_.view(1,1),B_prob_.view(1,1)]).view(-1,2))
                pred_train = torch.cat(prob_list,dim = 0)
                loss = self.loss_fn(pred_train,batch_label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        print ("[BIDAF]",end = "")
        return self
    def predict_proba(self,X_test):
        self.bidaf.eval()
        pred_bidaf = []
        batch_size = 1
        with torch.no_grad():
            for b in range(0,X_test.shape[0],batch_size):

                start = b
                end = min(X_test.shape[0],b+batch_size)

                batch_data = X_test[vector][start:end]
                batch_label = X_test.label[start:end]
                batch_pron = X_test[pron_vector][start:end]
                batch_pron = torch.Tensor(np.array(list(batch_pron.values))).unsqueeze(1).cuda()
                batch_data_torch = pad_sequence([torch.Tensor(v) for v in batch_data]).cuda().transpose(0,1)
                batch_padding = batch_data_torch.mean(dim=1,keepdim = True)#torch.zeros(batch_data.size()[0],1,batch_data.size()[2]).cuda()*0.001
                batch_data_torch = torch.cat([batch_padding,batch_data_torch],dim = 1)
                batch_label = torch.LongTensor(list(batch_label)).cuda()
                c_mask = torch.zeros_like(batch_data_torch.mean(-1,keepdim = True)) != batch_data_torch.mean(-1,keepdim = True)
                q_mask = torch.zeros_like(batch_pron.mean(-1,keepdim = True)) != batch_pron.mean(-1,keepdim = True)
                c_mask = c_mask.cuda()
                q_mask = q_mask.cuda()
                output = self.bidaf(batch_data_torch,batch_pron,c_mask,q_mask)
                mask_A = [np.array(v)+1 for v in list(X_test[A_idx][start:end])]
                mask_B = [np.array(v)+1 for v in list(X_test[B_idx][start:end])]
                mask_neither = [np.array(v)+1 for v in list(X_test[neither_idx][start:end])]
                prob_list = []
                for i,(v_A,v_B,v_neither) in enumerate(zip(mask_A,mask_B,mask_neither)):
                    v_A = torch.LongTensor(v_A).cuda()
                    A_prob_ = output[i,v_A].mean()
                    v_B = torch.LongTensor(v_B).cuda()
                    B_prob_ = output[i,v_B].mean()
                    v_neither = list(v_neither)
                    v_neither.append(0)
                    v_neither = torch.LongTensor(v_neither).cuda()
                    other_prob = output[i,v_neither].mean()
                    prob_list.append(torch.cat([A_prob_.view(1,1),B_prob_.view(1,1)]).view(-1,2))
                pred_bidaf_ = torch.cat(prob_list,dim = 0)
                pred_bidaf.append(pred_bidaf_)

        return torch.nn.Softmax(dim=1)(torch.cat(pred_bidaf,dim = 0)).cpu().data.numpy()
