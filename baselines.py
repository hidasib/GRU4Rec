# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015

@author: Hidasi Balázs
"""

import numpy as np
import pandas as pd

class RandomPred:

    def fit(self, a):
        pass

    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        return pd.Series(data=np.random.rand(len(predict_for_item_ids)), index=predict_for_item_ids)

class Pop:
    
    def __init__(self, top_n = 100, item_key = 'ItemId', support_by_key = None):
        self.top_n = top_n
        self.item_key = item_key
        self.support_by_key = support_by_key
    
    def fit(self, data):
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size() if self.support_by_key is None else grp[self.support_by_key].nunique()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)
        
    def predict_simple(self, session_id, predict_for_item_ids):
        preds = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, self.pop_list.index)
        preds[mask] = self.pop_list[predict_for_item_ids[mask]]
        return preds    
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        preds = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, self.pop_list.index)
        preds[mask] = self.pop_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)
        
class PersonalPop:
    
    def __init__(self, top_n = 100, item_key = 'ItemId', support_by_key = None):
        self.top_n = top_n
        self.item_key = item_key
        self.support_by_key = support_by_key
    
    def fit(self, data):
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size() if self.support_by_key is None else grp[self.support_by_key].nunique()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)
        self.prev_session_id = -1
         
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        if self.prev_session_id != session_id:
            self.prev_session_id = session_id
            self.pers = dict()
        v = self.pers.get(input_item_id)
        if v:
            self.pers[input_item_id] = v + 1
        else:
            self.pers[input_item_id] = 1
        preds = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, self.pop_list.index)
        ser = pd.Series(self.pers)
        preds[mask] = self.pop_list[predict_for_item_ids[mask]] 
        mask = np.in1d(predict_for_item_ids, ser.index)
        preds[mask] += ser[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)
 
class CoOccurence:
    
    def __init__(self, n_sims = 100, lmbd = 20, alpha = 0.5, normalize = True, seq_only = False, later_only = False, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time', assume_indexed = False):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.normalize = normalize
        self.seq_only = seq_only
        self.later_only = later_only
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key
        self.assume_indexed = assume_indexed

    def fit(self, data):
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        n_items = len(itemids)
        if not self.assume_indexed:   
            data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(len(itemids))}), on=self.item_key, how='inner')
            sessionids = data[self.session_key].unique()
            data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(len(sessionids))}), on=self.session_key, how='inner')
        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp)+1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values
        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items+1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values
        self.sims = dict()
        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i+1]
            for e in index_by_items[start:end]:
                uidx = data.SessionIdx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx+1]
                user_events = index_by_sessions[ustart:uend]
                if self.later_only:
                    mask = data[self.time_key].values[user_events] > data[self.time_key].values[e]
                    user_events = user_events[mask]
                elif self.seq_only:
                    mask = data[self.time_key].values[user_events] > data[self.time_key].values[e]
                    user_events = user_events[mask]
                    if len(user_events) > 0: user_events = user_events[0]
                iarray[data.ItemIdx.values[user_events]] += 1
            iarray[i] = 0
            if self.normalize:
                norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
                norm[norm == 0] = 1
                iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1-self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        preds = np.zeros(len(predict_for_item_ids))
        sim_list = self.sims[input_item_id]
        mask = np.in1d(predict_for_item_ids, sim_list.index)
        preds[mask] = sim_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)

class Cosine(CoOccurence):
    '''
    Speficication of general CoOcurence to be a simple cosine similarity between items.
    '''
    def __init__(self, n_sims = 100, lmbd = 20,  assume_indexed = False):
        self.n_sims = n_sims
        self.lmbd = lmbd
        #cosine makes: alpha = 0.5, normalize = True, seq_only = False,
        self.alpha = 0.5
        self.normalize = True
        self.seq_only = False
        self.later_only = False
        self.assume_indexed = assume_indexed

        self.item_key = 'ItemId'
        self.session_key = 'SessionId'
        self.time_key = 'Time'

class BPR:
    def __init__(self, n_factors = 80, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId', assume_indexed = False, pred_epochs = 10):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_session = lambda_session
        self.lambda_item = lambda_item
        self.sigma = sigma
        self.init_normal = init_normal
        self.session_key = session_key
        self.item_key = item_key
        self.assume_indexed = assume_indexed
        self.current_session = None
        self.pred_epochs = pred_epochs

    def init(self, data):
        self.U = np.random.rand(self.n_sessions, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_sessions, self.n_factors) * self.sigma
        self.I = np.random.rand(self.n_items, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_items, self.n_factors) * self.sigma
        self.bU = np.zeros(self.n_sessions)
        self.bI = np.zeros(self.n_items)
    
    def update(self, uidx, p, n):
        uF = np.copy(self.U[uidx,:])
        iF1 = np.copy(self.I[p,:])
        iF2 = np.copy(self.I[n,:])
        sigm = self.sigmoid(iF1.T.dot(uF) - iF2.T.dot(uF) + self.bI[p] - self.bI[n])
        c = 1.0 - sigm
        self.U[uidx,:] += self.learning_rate * (c * (iF1 - iF2) - self.lambda_session * uF)
        self.I[p,:] += self.learning_rate * (c * uF - self.lambda_item * iF1)
        self.I[n,:] += self.learning_rate * (-c * uF - self.lambda_item * iF2)
        return np.log(sigm)
    
    def fit(self, data):
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        if not self.assume_indexed:   
            data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(self.n_items)}), on=self.item_key, how='inner')
            data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(self.n_sessions)}), on=self.session_key, how='inner')     
        self.init(data)
        for it in range(self.n_iterations):
            c = []
            for e in np.random.permutation(len(data)):
                uidx = data.SessionIdx.values[e]
                iidx = data.ItemIdx.values[e]
                iidx2 = data.ItemIdx.values[np.random.randint(self.n_items)]
                err = self.update(uidx, iidx, iidx2)
                c.append(err)
            print(it, np.mean(c))
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):        
        iidx = self.itemidmap[input_item_id]
        if self.current_session is None or self.current_session != session_id:
            self.current_session = session_id
            self.session = [iidx]
        else:
            self.session.append(iidx)
        uF = self.I[self.session].mean(axis=0)
        iIdxs = self.itemidmap[predict_for_item_ids]
        return pd.Series(data=self.I[iIdxs].dot(uF) + self.bI[iIdxs], index=predict_for_item_ids)
             
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))