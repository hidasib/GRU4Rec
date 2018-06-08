# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:17:58 2017

@author: Balázs Hidasi
"""

import theano
from theano import tensor as T

def gpu_diag_wide(X):
    E = T.eye(*X.shape)
    return T.sum(X*E, axis=1)

def gpu_diag_tall(X):
    E = T.eye(*X.shape)
    return T.sum(X*E, axis=0)