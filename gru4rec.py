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
    '''
    GRU4Rec(loss, final_act, hidden_act, layers,
                 n_epochs=10, batch_size=50, dropout_p_hidden=0.5, dropout_p_embed=0.0, learning_rate=0.05, momentum=0.0, lmbd=0.0, embedding=0, n_sample=0, sample_alpha=0.75, smoothing=0,
                 adapt='adagrad', decay=0.9, grad_cap=0,
                 sigma=0, init_as_normal=False, reset_after_session=True, train_random_order=False, time_sort=True,
                 session_key='SessionId', item_key='ItemId', time_key='Time')
    Initializes the network.

    Parameters
    -----------
    loss : 'top1', 'bpr', 'cross-entropy', 'xe_logit', top1-max, bpr-max-<X>
        selects the loss function, <X> is the parameter of the loss
    final_act : 'softmax', 'linear', 'relu', 'tanh', 'softmax_logit', 'leaky-<X>', elu-<X>
        selects the activation function of the final layer, <X> is the parameter of the activation function
    hidden_act : 'tanh', 'relu' or 'linear'
        selects the activation function on the hidden states
    layers : 1D array
        list of the number of GRU units in the layers
    n_epochs : int
        number of training epochs (default: 10)
    batch_size : int
        size of the minibacth, also effect the number of negative samples through minibatch based sampling (default: 50)
    dropout_p_hidden : float
        probability of dropout of hidden units (default: 0.5)
    dropout_p_embed : float
        probability of dropout of the input units, applicable only if embeddings are used (default: 0.0)
    learning_rate : float
        learning rate (default: 0.05)
    momentum : float
        if not zero, Nesterov momentum will be applied during training with the given strength (default: 0.0)
    lmbd : float
        coefficient of the L2 regularization (default: 0.0)
    embedding : int
        size of the embedding used, 0 means not to use embedding (default: 0)
    n_sample : int
        number of additional negative samples to be used (besides the other examples of the minibatch) (default: 0)
    sample_alpha : float
        the probability of an item used as an additional negative sample is supp^sample_alpha (default: 0.75)
        (e.g.: sample_alpha=1 --> popularity based sampling; sample_alpha=0 --> uniform sampling)
    smoothing : float (default: 0)
        (only works with cross-entropy and xe_logit losses) if set to non-zero class labels are smoothed with this value, i.e. the expected utput is (e/N, ..., e/N, 1-e+e/N, e/N, ..., e/N) instead of (0, ..., 0, 1, 0, ..., 0), where N is the number of outputs and e is the smoothing value
    adapt : None, 'adagrad', 'rmsprop', 'adam', 'adadelta'
        sets the appropriate learning rate adaptation strategy, use None for standard SGD (default: 'adagrad')
    decay : float
        decay parameter for RMSProp, has no effect in other modes (default: 0.9)
    grad_cap : float
        clip gradients that exceede this value to this value, 0 means no clipping (default: 0.0)
    sigma : float
        "width" of initialization; either the standard deviation or the min/max of the init interval (with normal and uniform initializations respectively); 0 means adaptive normalization (sigma depends on the size of the weight matrix); (default: 0)
    init_as_normal : boolean
        False: init from uniform distribution on [-sigma,sigma]; True: init from normal distribution N(0,sigma); (default: False)
    reset_after_session : boolean
        whether the hidden state is set to zero after a session finished (default: True)
    train_random_order : boolean
        whether to randomize the order of sessions in each epoch (default: False)
    time_sort : boolean
        whether to ensure the the order of sessions is chronological (default: True)
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')

    '''
    def __init__(self, loss, final_act, hidden_act, layers,
                 n_epochs=10, batch_size=50, dropout_p_hidden=0.5, dropout_p_embed=0.0, learning_rate=0.05, momentum=0.0, lmbd=0.0, embedding=0, n_sample=0, sample_alpha=0.75, smoothing=0,
                 adapt='adagrad', decay=0.9, grad_cap=0,
                 sigma=0, init_as_normal=False, reset_after_session=True, train_random_order=False, time_sort=True,
                 session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.layers = layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_p_hidden = dropout_p_hidden
        self.dropout_p_embed = dropout_p_embed
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
        self.embedding = embedding
        self.time_sort = time_sort
        if adapt == 'rmsprop': self.adapt = 'rmsprop'
        elif adapt == 'adagrad': self.adapt = 'adagrad'
        elif adapt == 'adadelta': self.adapt = 'adadelta'
        elif adapt == 'adam': self.adapt = 'adam'
        else: self.adapt = False
        self.final_act = final_act
        if final_act == 'linear': self.final_activation = self.linear
        elif final_act == 'relu': self.final_activation = self.relu
        elif final_act == 'softmax': self.final_activation=self.softmax
        elif final_act == 'tanh': self.final_activation=self.tanh
        elif final_act == 'softmax_logit': self.final_activation=self.softmax_logit
        elif final_act.startswith('leaky-'):
            self.final_activation=self.leaky
            self.leak = float(final_act[6:])
        elif final_act.startswith('elu-'):
            self.final_activation=self.elu
            self.elu_param = float(final_act[4:])
        else: raise NotImplementedError
        self.loss = loss
        if loss == 'cross-entropy': self.loss_function = self.cross_entropy
        elif loss == 'bpr': self.loss_function = self.bpr
        elif loss.startswith('bpr-max-'): 
            self.loss_function = self.bpr_max
            self.bpreg = float(loss[8:])
            self.hack_matrix = np.ones((self.batch_size, self.batch_size + n_sample), dtype=np.float32)
            np.fill_diagonal(self.hack_matrix, 0)
            self.hack_matrix = theano.shared(self.hack_matrix, borrow=True)
        elif loss == 'top1': self.loss_function = self.top1
        elif loss == 'top1-max': 
            self.loss_function = self.top1_max
            self.hack_matrix = np.ones((self.batch_size, self.batch_size + n_sample), dtype=np.float32)
            np.fill_diagonal(self.hack_matrix, 0)
            self.hack_matrix = theano.shared(self.hack_matrix, borrow=True)
        elif loss == 'xe_logit': self.loss_function = self.cross_entropy_logits
        else: raise NotImplementedError
        self.hidden_act = hidden_act
        if hidden_act == 'relu': self.hidden_activation = self.relu
        elif hidden_act == 'tanh': self.hidden_activation = self.tanh
        elif hidden_act == 'linear': self.hidden_activation = self.linear
        else: raise NotImplementedError
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.smoothing = smoothing
    ######################ACTIVATION FUNCTIONS#####################
    def linear(self,X):
        return X
    def tanh(self,X):
        return T.tanh(X)
    def softmax(self,X):
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    def softmax_logit(self, X):
        X = X - X.max(axis=1).dimshuffle(0, 'x')
        return T.log(T.exp(X).sum(axis=1).dimshuffle(0, 'x')) - X
    def softmax_neg(self, X):
        if hasattr(self, 'hack_matrix'):
            X = X * self.hack_matrix
            e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x')) * self.hack_matrix
        else:
            e_x = T.fill_diagonal(T.exp(X - X.max(axis=1).dimshuffle(0, 'x')), 0)
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    def relu(self,X):
        return T.maximum(X, 0)
    def leaky(self,X):
        return T.switch(T.ge(X, 0), X, self.leak * X)
    def elu(self,X):
        return T.switch(T.ge(X, 0), X, self.elu_param * (T.exp(X) - 1))
    def sigmoid(self, X):
        return T.nnet.sigmoid(X)
    #################################LOSS FUNCTIONS################################
    def cross_entropy(self, yhat):
        if self.smoothing:
            n_out = self.batch_size + self.n_sample
            return T.cast(T.mean((1.0-(n_out/(n_out-1))*self.smoothing) * (-T.log(T.diag(yhat)+1e-24)) + (self.smoothing/(n_out-1)) * T.sum(-T.log(yhat+1e-24), axis=1)), theano.config.floatX)
        else:
            return T.cast(T.mean(-T.log(T.diag(yhat)+1e-24)), theano.config.floatX)
    def cross_entropy_logits(self, yhat):
        if self.smoothing:
            n_out = self.batch_size + self.n_sample
            return T.cast(T.mean((1.0-(n_out/(n_out-1))*self.smoothing) * T.diag(yhat) + (self.smoothing/(n_out-1)) * T.sum(yhat, axis=1)), theano.config.floatX)
        else:
            return T.cast(T.mean(T.diag(yhat)), theano.config.floatX)
    def bpr(self, yhat):
        return T.cast(T.mean(-T.log(T.nnet.sigmoid(T.diag(yhat)-yhat.T))), theano.config.floatX)
    def bpr_max(self, yhat):
        softmax_scores = self.softmax_neg(yhat).T
        return T.cast(T.mean(-T.log(T.sum(T.nnet.sigmoid(T.diag(yhat)-yhat.T)*softmax_scores, axis=0)+1e-24)+self.bpreg*T.sum((yhat.T**2)*softmax_scores, axis=0)), theano.config.floatX)
    def top1(self, yhat):
        yhatT = yhat.T
        return T.cast(T.mean(T.mean(T.nnet.sigmoid(-T.diag(yhat)+yhatT)+T.nnet.sigmoid(yhatT**2), axis=0)-T.nnet.sigmoid(T.diag(yhat)**2)/(self.batch_size+self.n_sample)), theano.config.floatX)
    def top1_max(self, yhat):
        yhatT = yhat.T
        softmax_scores = self.softmax_neg(yhat)      
        y = softmax_scores.T*(T.nnet.sigmoid(-T.diag(yhat)+yhatT)+T.nnet.sigmoid(yhatT**2))      
        return T.cast(T.mean(T.sum(y, axis=0)), theano.config.floatX)
    ###############################################################################
    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)
    def init_weights(self, shape):
        return theano.shared(self.init_matrix(shape), borrow=True)
    def init_matrix(self, shape):
        if self.sigma != 0: sigma = self.sigma
        else: sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        if self.init_as_normal:
            return self.floatX(np.random.randn(*shape) * sigma)
        else:
            return self.floatX(np.random.rand(*shape) * sigma * 2 - sigma)
    def extend_weights(self, W, n_new):
        matrix = W.get_value()
        sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (matrix.shape[0] + matrix.shape[1] + n_new))
        if self.init_as_normal: new_rows = self.floatX(np.random.randn(n_new, matrix.shape[1]) * sigma)
        else: new_rows = self.floatX(np.random.rand(n_new, matrix.shape[1]) * sigma * 2 - sigma)
        W.set_value(np.vstack([matrix, new_rows]))
    def init(self, data):
        data.sort_values([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        np.random.seed(42)
        self.Wx, self.Wh, self.Wrz, self.Bh, self.H = [], [], [], [], []
        if self.embedding:
            self.E = self.init_weights((self.n_items, self.embedding))
            n_features = self.embedding
        else:
            n_features = self.n_items
        for i in range(len(self.layers)):
            m = []
            m.append(self.init_matrix((self.layers[i-1] if i > 0 else n_features, self.layers[i])))
            m.append(self.init_matrix((self.layers[i-1] if i > 0 else n_features, self.layers[i])))
            m.append(self.init_matrix((self.layers[i-1] if i > 0 else n_features, self.layers[i])))
            self.Wx.append(theano.shared(value=np.hstack(m), borrow=True))
            self.Wh.append(self.init_weights((self.layers[i], self.layers[i])))
            m2 = []
            m2.append(self.init_matrix((self.layers[i], self.layers[i])))
            m2.append(self.init_matrix((self.layers[i], self.layers[i])))
            self.Wrz.append(theano.shared(value=np.hstack(m2), borrow=True))
            self.Bh.append(theano.shared(value=np.zeros((self.layers[i] * 3,), dtype=theano.config.floatX), borrow=True))
            self.H.append(theano.shared(value=np.zeros((self.batch_size,self.layers[i]), dtype=theano.config.floatX), borrow=True))
        self.Wy = self.init_weights((self.n_items, self.layers[-1]))
        self.By = theano.shared(value=np.zeros((self.n_items,1), dtype=theano.config.floatX), borrow=True)
        return offset_sessions
    def dropout(self, X, drop_p):
        if drop_p > 0:
            retain_prob = 1 - drop_p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX) / retain_prob
        return X
    def adam(self, param, grad, updates, sample_idx = None, epsilon = 1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        meang = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        countt = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = v1 * acc + v2 * grad ** 2
            meang_new = v1 * meang + v2 * grad
            countt_new = countt + 1
            updates[acc] = acc_new
            updates[meang] = meang_new
            updates[countt] = countt_new
        else:
            acc_s = acc[sample_idx]
            meang_s = meang[sample_idx]
            countt_s = countt[sample_idx]
            acc_new = v1 * acc_s + v2 * grad ** 2
            meang_new = v1 * meang_s + v2 * grad
            countt_new = countt_s + 1.0
            updates[acc] = T.set_subtensor(acc_s, acc_new)
            updates[meang] = T.set_subtensor(meang_s, meang_new)
            updates[countt] = T.set_subtensor(countt_s, countt_new)
        return (meang_new / (1 - v1**countt_new)) / (T.sqrt(acc_new / (1 - v1**countt_new)) + epsilon)
    def adagrad(self, param, grad, updates, sample_idx = None, epsilon = 1e-6):
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = acc + grad ** 2
            updates[acc] = acc_new
        else:
            acc_s = acc[sample_idx]
            acc_new = acc_s + grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling
    def adadelta(self, param, grad, updates, sample_idx = None, epsilon = 1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        upd = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = acc + grad ** 2
            updates[acc] = acc_new
            grad = T.sqrt(upd + epsilon) * grad
            upd_new = v1 * upd + v2 * grad ** 2
            updates[upd] = upd_new
        else:
            acc_s = acc[sample_idx]
            acc_new = acc_s + grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
            upd_s = upd[sample_idx]
            upd_new = v1 * upd_s + v2 * grad ** 2
            updates[upd] = T.set_subtensor(upd_s, upd_new)
            grad = T.sqrt(upd_s + epsilon) * grad
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling
    def rmsprop(self, param, grad, updates, sample_idx = None, epsilon = 1e-6):
        v1 = np.float32(self.decay)
        v2 = np.float32(1.0 - self.decay)
        acc = theano.shared(param.get_value(borrow=False) * 0., borrow=True)
        if sample_idx is None:
            acc_new = v1 * acc + v2 * grad ** 2
            updates[acc] = acc_new
        else:
            acc_s = acc[sample_idx]
            acc_new = v1 * acc_s + v2 * grad ** 2
            updates[acc] = T.set_subtensor(acc_s, acc_new)
        gradient_scaling = T.cast(T.sqrt(acc_new + epsilon), theano.config.floatX)
        return grad / gradient_scaling
    def RMSprop(self, cost, params, full_params, sampled_params, sidxs, epsilon=1e-6):
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
                    if self.adapt == 'adagrad':
                        g = self.adagrad(p, g, updates)
                    if self.adapt == 'rmsprop':
                        g = self.rmsprop(p, g, updates)
                    if self.adapt == 'adadelta':
                        g = self.adadelta(p, g, updates)
                    if self.adapt == 'adam':
                        g = self.adam(p, g, updates)
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
                if self.adapt == 'adagrad':
                    g = self.adagrad(fullP, g, updates, sample_idx)
                if self.adapt == 'rmsprop':
                    g = self.rmsprop(fullP, g, updates, sample_idx)
                if self.adapt == 'adadelta':
                    g = self.adadelta(fullP, g, updates, sample_idx)
                if self.adapt == 'adam':
                    g = self.adam(fullP, g, updates, sample_idx)
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
    def model(self, X, H, Y=None, drop_p_hidden=0.0, drop_p_embed=0.0, predict=False):
        if self.embedding:
            Sx = self.E[X]
            y = self.dropout(Sx, drop_p_embed)
            H_new = []
            start = 0
        else:
            Sx = self.Wx[0][X]
            vec = Sx + self.Bh[0]
            rz = T.nnet.sigmoid(vec.T[self.layers[0]:] + T.dot(H[0], self.Wrz[0]).T)
            h = self.hidden_activation(T.dot(H[0] * rz[:self.layers[0]].T, self.Wh[0]) + vec.T[:self.layers[0]].T)
            z = rz[self.layers[0]:].T
            h = (1.0-z)*H[0] + z*h
            h = self.dropout(h, drop_p_hidden)
            H_new = [h]
            y = h
            start = 1
        for i in range(start, len(self.layers)):
            vec = T.dot(y, self.Wx[i]) + self.Bh[i]
            rz = T.nnet.sigmoid(vec.T[self.layers[i]:] + T.dot(H[i], self.Wrz[i]).T)
            h = self.hidden_activation(T.dot(H[i] * rz[:self.layers[i]].T, self.Wh[i]) + vec.T[:self.layers[i]].T)
            z = rz[self.layers[i]:].T
            h = (1.0-z)*H[i] + z*h
            h = self.dropout(h, drop_p_hidden)
            H_new.append(h)
            y = h
        if Y is not None:
            Sy = self.Wy[Y]
            SBy = self.By[Y]
            if predict and self.final_act == 'softmax_logit':
                y = self.softmax(T.dot(y, Sy.T) + SBy.flatten())
            else:
                y = self.final_activation(T.dot(y, Sy.T) + SBy.flatten())
            return H_new, y, [Sx, Sy, SBy]
        else:
            if predict and self.final_act == 'softmax_logit':
                y = self.softmax(T.dot(y, self.Wy.T) + self.By.flatten())
            else:
                y = self.final_activation(T.dot(y, self.Wy.T) + self.By.flatten())
            return H_new, y, [Sx]
    def generate_neg_samples(self, pop, length):
        if self.sample_alpha:
            sample = np.searchsorted(pop, np.random.rand(self.n_sample *  length))
        else:
            sample = np.random.choice(self.n_items, size=self.n_sample * length)
        if length > 1:
            sample = sample.reshape((length, self.n_sample))
        return sample
    def fit(self, data, retrain=False, sample_store=10000000):
        '''
        Trains the network.

        Parameters
        --------
        data : pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        retrain : boolean
            If False, do normal train. If True, do additional train (weigths from previous trainings are kept as the initial network) (default: False)
        sample_store : int
            If additional negative samples are used (n_sample > 0), the efficiency of GPU utilization can be sped up, by precomputing a large batch of negative samples (and recomputing when necessary).
            This parameter regulizes the size of this precomputed ID set. Its value is the maximum number of int values (IDs) to be stored. Precomputed IDs are stored in the RAM.
            For the most efficient computation, a balance must be found between storing few examples and constantly interrupting GPU computations for a short time vs. computing many examples and interrupting GPU computations for a long time (but rarely).

        '''
        self.predict = None
        self.error_during_train = False
        itemids = data[self.item_key].unique()
        if not retrain:
            self.n_items = len(itemids)
            self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
            data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':self.itemidmap[itemids].values}), on=self.item_key, how='inner')
            offset_sessions = self.init(data)
        else:
            new_item_mask = ~np.in1d(itemids, self.itemidmap.index)
            n_new_items = new_item_mask.sum()
            if n_new_items:
                self.itemidmap = self.itemidmap.append(pd.Series(index=itemids[new_item_mask], data=np.arange(n_new_items) + len(self.itemidmap)))
                for W in [self.E if self.embedding else self.Wx[0], self.Wy]:
                    self.extend_weights(W, n_new_items)
                self.By.set_value(np.vstack([self.By.get_value(), np.zeros((n_new_items, 1), dtype=theano.config.floatX)]))
                self.n_items += n_new_items
                print('Added {} new items. Number of items is {}.'.format(n_new_items, self.n_items))
            data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':self.itemidmap[itemids].values}), on=self.item_key, how='inner')
            data.sort_values([self.session_key, self.time_key], inplace=True)
            offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
            offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        X = T.ivector()
        Y = T.ivector()
        H_new, Y_pred, sampled_params = self.model(X, self.H, Y, self.dropout_p_hidden, self.dropout_p_embed)
        cost = self.loss_function(Y_pred)
        params = [self.Wx if self.embedding else self.Wx[1:], self.Wh, self.Wrz, self.Bh]
        full_params = [self.E if self.embedding else self.Wx[0], self.Wy, self.By]
        sidxs = [X, Y, Y]
        updates = self.RMSprop(cost, params, full_params, sampled_params, sidxs)
        for i in range(len(self.H)):
            updates[self.H[i]] = H_new[i]
        train_function = function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        base_order = np.argsort(data.groupby(self.session_key)[self.time_key].min().values) if self.time_sort else np.arange(len(offset_sessions)-1)
        if self.n_sample:
            pop = data.groupby('ItemId').size()
            pop = pop[self.itemidmap.index.values].values**self.sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            if sample_store:
                generate_length = sample_store // self.n_sample
                if generate_length <= 1:
                    sample_store = 0
                    print('No example store was used')
                else:
                    neg_samples = self.generate_neg_samples(pop, generate_length)
                    sample_pointer = 0
                    print('Created sample store with {} batches of samples'.format(generate_length))
            else:
                print('No example store was used')
        data_items = data.ItemIdx.values
        for epoch in range(self.n_epochs):
            for i in range(len(self.layers)):
                self.H[i].set_value(np.zeros((self.batch_size,self.layers[i]), dtype=theano.config.floatX), borrow=True)
            c = []
            session_idx_arr = np.random.permutation(len(offset_sessions)-1) if self.train_random_order else base_order
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters]+1]
            finished = False
            while not finished:
                minlen = (end-start).min()
                out_idx = data_items[start]
                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx = data_items[start+i+1]
                    if self.n_sample:
                        if sample_store:
                            if sample_pointer == generate_length:
                                neg_samples = self.generate_neg_samples(pop, generate_length)
                                sample_pointer = 0
                            sample = neg_samples[sample_pointer]
                            sample_pointer += 1
                        else:
                            sample = self.generate_neg_samples(pop, 1)
                        y = np.hstack([out_idx, sample])
                    else:
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
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter]+1]
                if len(mask) and self.reset_after_session:
                    for i in range(len(self.H)):
                        tmp = self.H[i].get_value(borrow=True)
                        tmp[mask] = 0
                        self.H[i].set_value(tmp, borrow=True)
            avgc = np.mean(c)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))
    def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        if self.error_during_train: raise Exception
        if self.predict is None or self.predict_batch!=batch:
            X = T.ivector()
            Y = T.ivector()
            for i in range(len(self.layers)):
                self.H[i].set_value(np.zeros((batch,self.layers[i]), dtype=theano.config.floatX), borrow=True)
            if predict_for_item_ids is not None:
                H_new, yhat, _ = self.model(X, self.H, Y, predict=True)
            else:
                H_new, yhat, _ = self.model(X, self.H, predict=True)
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
