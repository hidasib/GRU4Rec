# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015

@author: Balázs Hidasi
"""

import numpy as np
import pandas as pd

class RandomPred:
    '''
    RandomPred()
    
    Initializes a random predcitor, which is a baseline predictor that gives back a random score for each item.  
    
    '''
    def fit(self, data):
        '''
        Dummy function for training.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        pass

    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        return pd.Series(data=np.random.rand(len(predict_for_item_ids)), index=predict_for_item_ids)

class Pop:
    '''
    Pop(top_n=100, item_key='ItemId', support_by_key=None)
    
    Popularity predictor that gives higher scores to items with larger support.
    
    The score is given by:
    
    .. math::
        r_{i}=\\frac{supp_i}{(1+supp_i)}
        
    Parameters
    --------
    top_n : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    support_by_key : string or None
        If not None, count the number of unique values of the attribute of the training data given by the specified header. If None, count the events. (Default value: None)
    
    '''
    
    def __init__(self, top_n = 100, item_key = 'ItemId', support_by_key = None):
        self.top_n = top_n
        self.item_key = item_key
        self.support_by_key = support_by_key
    
    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size() if self.support_by_key is None else grp[self.support_by_key].nunique()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)  
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        preds = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, self.pop_list.index)
        preds[mask] = self.pop_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)
        
class SessionPop:
    '''
    SessionPop(top_n=100, item_key='ItemId', support_by_key=None)
    
    Session popularity predictor that gives higher scores to items with higher number of occurrences in the session. Ties are broken up by adding the popularity score of the item.
    
    The score is given by:
    
    .. math::
        r_{s,i} = supp_{s,i} + \\frac{supp_i}{(1+supp_i)}
        
    Parameters
    --------
    top_n : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    support_by_key : string or None
        If not None, count the number of unique values of the attribute of the training data given by the specified header. If None, count the events. (Default value: None)
    
    '''    
    def __init__(self, top_n = 100, item_key = 'ItemId', support_by_key = None):
        self.top_n = top_n
        self.item_key = item_key
        self.support_by_key = support_by_key
    
    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size() if self.support_by_key is None else grp[self.support_by_key].nunique()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)
        self.prev_session_id = -1
         
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event. If changed during subsequent calls, a new session starts.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
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
 
class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')
    
    Item-to-item predictor that computes the the similarity to all items to the given item.
    
    Similarity of two items is given by:
    
    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}
        
    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    
    '''    
    
    def __init__(self, n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key

    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        n_items = len(itemids) 
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
                iarray[data.ItemIdx.values[user_events]] += 1
            iarray[i] = 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1-self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        preds = np.zeros(len(predict_for_item_ids))
        sim_list = self.sims[input_item_id]
        mask = np.in1d(predict_for_item_ids, sim_list.index)
        preds[mask] = sim_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)

class BPR:
    '''
    BPR(n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId')
    
    Bayesian Personalized Ranking Matrix Factorization (BPR-MF). During prediction time, the current state of the session is modelled as the average of the feature vectors of the items that have occurred in it so far.
        
    Parameters
    --------
    n_factor : int
        The number of features in a feature vector. (Default value: 100)
    n_iterations : int
        The number of epoch for training. (Default value: 10)
    learning_rate : float
        Learning rate. (Default value: 0.01)
    lambda_session : float
        Regularization for session features. (Default value: 0.0)
    lambda_item : float
        Regularization for item features. (Default value: 0.0)
    sigma : float
        The width of the initialization. (Default value: 0.05)
    init_normal : boolean
        Whether to use uniform or normal distribution based initialization.
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    
    '''
    def __init__(self, n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId'):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_session = lambda_session
        self.lambda_item = lambda_item
        self.sigma = sigma
        self.init_normal = init_normal
        self.session_key = session_key
        self.item_key = item_key
        self.current_session = None

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
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
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
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''      
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