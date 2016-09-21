# GRU4Rec

Basic implementation of the algorithm of "Session-based Recommendations With Recurrent Neural Networks". See paper: http://arxiv.org/abs/1511.06939


## Requirements

The code is written in Theano under python 3.4. For efficient execution on GPUs, Theano must include the updated version of the GPUIncAdvancedSubtensor1 operator. It is available in Theano-0.8.0.


## Update 21-09-2016
- Optimized code for GPU execution. Training is ~2x faster now.
- Added retrain functionality.