# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:14:20 2015
@author: Balázs Hidasi
"""

import theano
from theano import tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import pandas as pd
from collections import OrderedDict
srng = RandomStreams()
class GRU4Rec:
    ''' Initializes the network. You can set the following parameters.
    layers -- list of the number of GRU units in the layers (default: [100] --> 100 units in one layer)
    n_epochs -- number of training epochs (default: 10)
    batch_size -- size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 50)
    dropout_p_hidden -- probability of dropout of hidden units (default: 0.4)
    learning_rate -- learning rate (default: 0.05)
    momentum -- if not zero, Nesterov momentum will be applied during training with the given strength (default: 0.0)
    adapt -- either None, 'adagrad' or 'rmsprop', sets the appropriate learning rate adaptation strategy (default: 'adagrad')
    decay -- decay parameter for RMSProp, has no effect in other modes (default: 0.9)
    grad_cap -- clip gradients that exceede this value to this value, 0 means no clipping (default: 0.0)
    sigma -- "width" of initialization; either the standard deviation or the min/max of the init interval (with normal and uniform initializations respectively); 0 means adaptive normalization (sigma depends on the size of the weight matrix); (default: 0)
    init_as_normal -- False: init from uniform distribution on [-sigma,sigma]; True: init from normal distribution N(0,sigma); (default: False)
    reset_after_session -- whether the hidden state is set to zero after a session finished (default: True)
    loss -- 'top1', 'bpr' or 'cross-entropy' to select the loss function (default: 'top1')
    hidden_act -- 'tanh' or 'relu' to set the activation function on the hidden state (default: 'tanh')
    final_act -- None, 'linear', 'relu' or 'tanh' to set the activation function of the final layer where appropriate (cross-entropy always uses softmax), None means default (tanh if the loss is brp or top1) (default: None)
    train_random_order -- whether to randomize the order of sessions in each epoch (default: False)
    lmbd -- coefficient of the L2 regularization (default: 0.0)
    session_key -- header of the session ID column in the input file (default: 'SessionId')
    item_key -- header of the item ID column in the input file (default: 'ItemId')
    time_key -- header of the timestamp column in the input file (default: 'Time')
    '''
    def __init__(self, layers, n_epochs=10, batch_size=50, dropout_p_hidden=0.4, learning_rate=0.05, momentum=0.0, adapt='adagrad', decay=0.9, grad_cap=0, sigma=0, 
                 init_as_normal=False, reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False, lmbd=0.0, 
                 session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.layers = layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.sigma = sigma
        self.init_as_normal = init_as_normal
        self.reset_after_session = reset_after_session
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.grad_cap = grad_cap
        self.train_random_order = train_random_order
        self.lmbd = lmbd
        if adapt == 'rmsprop':
            self.adapt = 'rmsprop'
        elif adapt == 'adagrad':
            self.adapt = 'adagrad'
        else:
            self.adapt = False
        if loss=='cross-entropy':
            self.final_activation=self.softmax
            self.loss_function=self.cross_entropy
        elif loss=='bpr':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation=self.tanh
            self.loss_function=self.bpr
        elif loss=='top1':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation=self.tanh
            self.loss_function=self.top1
        else:
            raise NotImplementedError
        if hidden_act=='relu':
            self.hidden_activation=self.relu
        elif hidden_act=='tanh':
            self.hidden_activation=self.tanh
        else:
            raise NotImplementedError
    ######################ACTIVATION FUNCTIONS FOR FINAL LAYER#####################
    def linear(self,X):
        return X
    def tanh(self,X):
        return T.tanh(X)
    def softmax(self,X):
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    def relu(self,X):
        return T.maximum(X, 0)
    def sigmoid(self, X):
        return T.nnet.sigmoid(X)
    #################################LOSS FUNCTIONS################################
    #It is assumed that only one desired output is active (otherwise it will be very slow)
    #Use softmax activation function as the output with cross-entropy
    def cross_entropy(self, yhat):
        return T.cast(T.mean(-T.log(T.diag(yhat))), theano.config.floatX)
    #It is assumed that only one desired output is active (otherwise it will be very slow)
    #Use linear or tanh activation function as the output with BPR
    def bpr(self, yhat):
        return T.cast(T.mean(-T.log(T.nnet.sigmoid(T.diag(yhat)-yhat.T))), theano.config.floatX)
    #It is assumed that only one desired output is active (otherwise it will be very slow)
    #Use linear or tanh activation function as the output with error-rate
    def top1(self, yhat):
        return T.cast(T.mean(T.mean(T.nnet.sigmoid(-T.diag(yhat)+yhat.T)+T.nnet.sigmoid(yhat.T**2), axis=0)-T.nnet.sigmoid(T.diag(yhat)**2)/self.batch_size), theano.config.floatX)
    
    ###############################################################################
    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)
    def init_weights(self, shape):
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (shape[0] + shape[1]))
        if self.init_as_normal:
            return theano.shared(self.floatX(np.random.randn(*shape) * sigma), borrow=True)
        else:
            return theano.shared(self.floatX(np.random.rand(*shape) * sigma * 2 - sigma), borrow=True)
    def init(self, data):
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        np.random.seed(42)
        self.Wx, self.Wh, self.Wr, self.Wz, self.Whr, self.Whz, self.Bh, self.Br, self.Bz, self.H = [], [], [], [], [], [], [], [], [], []
        for i in range(len(self.layers)):
            self.Wx.append(self.init_weights((self.layers[i-1] if i > 0 else self.n_items, self.layers[i])))
            self.Wr.append(self.init_weights((self.layers[i-1] if i > 0 else self.n_items, self.layers[i])))
            self.Wz.append(self.init_weights((self.layers[i-1] if i > 0 else self.n_items, self.layers[i])))
            self.Wh.append(self.init_weights((self.layers[i], self.layers[i])))
            self.Whr.append(self.init_weights((self.layers[i], self.layers[i])))
            self.Whz.append(self.init_weights((self.layers[i], self.layers[i])))
            self.Bh.append(theano.shared(value=np.zeros((self.layers[i],), dtype=theano.config.floatX), borrow=True))
            self.Br.append(theano.shared(value=np.zeros((self.layers[i],), dtype=theano.config.floatX), borrow=True))
            self.Bz.append(theano.shared(value=np.zeros((self.layers[i],), dtype=theano.config.floatX), borrow=True))
            self.H.append(theano.shared(value=np.zeros((self.batch_size,self.layers[i]), dtype=theano.config.floatX), borrow=True))
        self.Wy = self.init_weights((self.n_items, self.layers[-1]))
        self.By = theano.shared(value=np.zeros((self.n_items,1), dtype=theano.config.floatX), borrow=True)
        return offset_sessions
    def dropout(self, X, drop_p):
        if drop_p > 0:
            retain_prob = 1 - drop_p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX) / retain_prob
        return X
    def RMSprop(self, cost, params, full_params, sampled_params, sidxs, epsilon=1e-6):
        v1 = np.float32(self.decay if self.adapt == 'rmsprop' else 1)
        v2 = np.float32((1.0 - self.decay) if self.adapt == 'rmsprop' else 1)
        grads =  [T.grad(cost = cost, wrt = param) for param in params]
        sgrads = [T.grad(cost = cost, wrt = sparam) for sparam in sampled_params]
        updates = OrderedDict()
        if self.grad_cap>0:
            norm=T.cast(T.sqrt(T.sum([T.sum([T.sum(g**2) for g in g_list]) for g_list in grads]) + T.sum([T.sum(g**2) for g in sgrads])), theano.config.floatX)
            grads = [[T.switch(T.ge(norm, self.grad_cap), g*self.grad_cap/norm, g) for g in g_list] for g_list in grads]
            sgrads = [T.switch(T.ge(norm, self.grad_cap), g*self.grad_cap/norm, g) for g in sgrads]
        for p_list, g_list in zip(params, grads):
            for p, g in zip(p_list, g_list):
                if self.adapt:
                    acc = theano.shared(p.get_value(borrow=False) * 0., borrow=True)
                    acc_new = v1 * acc + v2 * g ** 2
                    gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
                    g = g / gradient_scaling
                    updates[acc] = acc_new
                if self.momentum > 0:
                    velocity = theano.shared(p.get_value(borrow=False) * 0., borrow=True)
                    velocity2 = self.momentum * velocity - np.float32(self.learning_rate) * (g + self.lmbd * p)
                    updates[velocity] = velocity2
                    updates[p] = p + velocity2
                else:
                    updates[p] = p * np.float32(1.0 - self.learning_rate * self.lmbd) - np.float32(self.learning_rate) * g
        for i in range(len(sgrads)):
            g = sgrads[i]
            fullP = full_params[i]
            sample_idx = sidxs[i]
            sparam = sampled_params[i]
            if self.adapt:
                acc = theano.shared(fullP.get_value(borrow=False) * 0., borrow=True)
                acc_s = acc[sample_idx]
                acc_new = v1 * acc_s + v2 * g ** 2
                gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
                g = g / gradient_scaling
                updates[acc] = T.set_subtensor(acc_s, acc_new)
            if self.lmbd > 0:
                delta = np.float32(self.learning_rate) * (g + self.lmbd * sparam)
            else:
                delta = np.float32(self.learning_rate) * g
            if self.momentum > 0:
                velocity = theano.shared(fullP.get_value(borrow=False) * 0., borrow=True)
                vs = velocity[sample_idx]
                velocity2 = self.momentum * vs - delta
                updates[velocity] = T.set_subtensor(vs, velocity2)
                updates[fullP] = T.inc_subtensor(sparam, velocity2)
            else:
                updates[fullP] = T.inc_subtensor(sparam, - delta)
        return updates
    def model(self, X, H, Y, drop_p_hidden):
        Sx = self.Wx[0][X]
        Sr = self.Wr[0][X]
        Sz = self.Wz[0][X]
        r = T.nnet.sigmoid(Sr + T.dot(H[0], self.Whr[0]) + self.Br[0])
        h = self.hidden_activation(Sx + T.dot(H[0] * r, self.Wh[0]) + self.Bh[0])
        z = T.nnet.sigmoid(Sz + T.dot(H[0], self.Whz[0]) + self.Bz[0])
        h = (1.0-z)*H[0] + z*h
        h = self.dropout(h, drop_p_hidden)
        H_new = [h]
        y = h
        for i in range(1, len(self.layers)):
            r = T.nnet.sigmoid(T.dot(y, self.Wr[i]) + T.dot(H[i], self.Whr[i]) + self.Br[i])
            h = self.hidden_activation(T.dot(y, self.Wx[i]) + T.dot(H[i]*r, self.Wh[i]) + self.Bh[i])
            z = T.nnet.sigmoid(T.dot(y, self.Wz[i]) + T.dot(H[i], self.Whz[i]) + self.Bz[i])
            h = (1.0-z)*H[i] + z*h
            h = self.dropout(h, drop_p_hidden)
            H_new.append(h)
            y = h
        Sy = self.Wy[Y]
        SBy = self.By[Y]
        y = self.final_activation(T.dot(y, Sy.T) + SBy.flatten())
        return H_new, y, [Sx, Sr, Sz, Sy, SBy]
    def model_test(self, X, H):
        Sx = self.Wx[0][X]
        Sr = self.Wr[0][X]
        Sz = self.Wz[0][X]
        r = T.nnet.sigmoid(Sr + T.dot(H[0], self.Whr[0]) + self.Br[0])
        h = self.hidden_activation(Sx + T.dot(H[0] * r, self.Wh[0]) + self.Bh[0])
        z = T.nnet.sigmoid(Sz + T.dot(H[0], self.Whz[0]) + self.Bz[0])
        h = (1.0-z)*H[0] + z*h
        H_new = [h]
        y = h
        for i in range(1, len(self.layers)):
            r = T.nnet.sigmoid(T.dot(y, self.Wr[i]) + T.dot(H[i], self.Whr[i]) + self.Br[i])
            h = self.hidden_activation(T.dot(y, self.Wx[i]) + T.dot(H[i]*r, self.Wh[i]) + self.Bh[i])
            z = T.nnet.sigmoid(T.dot(y, self.Wz[i]) + T.dot(H[i], self.Whz[i]) + self.Bz[i])
            h = (1.0-z)*H[i] + z*h
            H_new.append(h)
            y = h
        y = self.final_activation(T.dot(y, self.Wy.T) + self.By.flatten())
        return H_new, y
    def fit(self, data):
        self.predict = None
        self.error_during_train = False
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':self.itemidmap[itemids].values}), on=self.item_key, how='inner')
        offset_sessions = self.init(data)
        
        X = T.ivector()
        Y = T.ivector()
        H_new, Y_pred, sampled_params = self.model(X, self.H, Y, self.dropout_p_hidden)
        cost = self.loss_function(Y_pred) 
        params = [self.Wx[1:], self.Wr[1:], self.Wz[1:], self.Wh, self.Whr, self.Whz, self.Bh, self.Br, self.Bz]
        full_params = [self.Wx[0], self.Wr[0], self.Wz[0], self.Wy, self.By]
        sidxs = [X, X, X, Y, Y]
        updates = self.RMSprop(cost, params, full_params, sampled_params, sidxs)  
        for i in range(len(self.H)):
            updates[self.H[i]] = H_new[i]
        train_function = function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        
        
        for epoch in range(self.n_epochs):
            for i in range(len(self.layers)):
                self.H[i].set_value(np.zeros((self.batch_size,self.layers[i]), dtype=theano.config.floatX), borrow=True)
            c = []
            session_idx_arr = np.random.permutation(len(offset_sessions)-1) if self.train_random_order else np.arange(len(offset_sessions)-1)
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters]+1]
            finished = False
            while not finished:
                minlen = (end-start).min()
                out_idx = data.ItemIdx.values[start]
                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx = data.ItemIdx.values[start+i+1]
                    y = out_idx
                    cost = train_function(in_idx, y)
                    c.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ': NaN error!')
                        self.error_during_train = True
                        return
                start = start+minlen-1
                mask = np.arange(len(iters))[(end-start)<=1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions)-1:
                        finished = True
                        break
                    if self.reset_after_session:
                        for i in range(len(self.H)):
                            tmp = self.H[i].get_value(borrow=True, return_internal_type=True)
                            tmp[idx,:] = 0
                            self.H[i].set_value(tmp, borrow=True)
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter]+1]
            avgc = np.mean(c)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))

    def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100):
        if self.error_during_train: raise Exception
        if self.predict is None or self.predict_batch!=batch:
            X = T.ivector()
            Y = T.ivector()
            for i in range(len(self.layers)):
                self.H[i].set_value(np.zeros((batch,self.layers[i]), dtype=theano.config.floatX), borrow=True)
            if predict_for_item_ids is not None:
                H_new, yhat, _ = self.model(X, self.H, Y, 0)
            else:
                H_new, yhat = self.model_test(X, self.H)
            updatesH = OrderedDict()
            for i in range(len(self.H)):
                updatesH[self.H[i]] = H_new[i]
            if predict_for_item_ids is not None:
                self.predict = function(inputs=[X, Y], outputs=yhat, updates=updatesH, allow_input_downcast=True)
            else:
                self.predict = function(inputs=[X], outputs=yhat, updates=updatesH, allow_input_downcast=True)
            self.current_session = np.ones(batch) * -1
            self.predict_batch = batch
        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:
            for i in range(len(self.H)):
                tmp = self.H[i].get_value(borrow=True)
                tmp[session_change] = 0
                self.H[i].set_value(tmp, borrow=True)
            self.current_session=session_ids.copy()
        in_idxs = self.itemidmap[input_item_ids]
        if predict_for_item_ids is not None:
            iIdxs = self.itemidmap[predict_for_item_ids]
            preds = np.asarray(self.predict(in_idxs, iIdxs)).T
            return pd.DataFrame(data=preds, index=predict_for_item_ids)
        else:
            preds = np.asarray(self.predict(in_idxs)).T
            return pd.DataFrame(data=preds, index=self.itemidmap.index)
