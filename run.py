import argparse
import os

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        try:
            columns = int(os.popen('stty size', 'r').read().split()[1])
        except:
            columns = None
        if columns is not None:
            self._width = columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train or load a GRU4Rec model & measure recall and MRR on the specified test set(s).')
parser.add_argument('path', metavar='PATH', type=str, help='Path to the training data (TAB separated file (.tsv or .txt) or pickled pandas.DataFrame object (.pickle)) (if the --load_model parameter is NOT provided) or to the serialized model (if the --load_model parameter is provided).')
parser.add_argument('-ps', '--parameter_string', metavar='PARAM_STRING', type=str, help='Training parameters provided as a single parameter string. The format of the string is `param_name1=param_value1,param_name2=param_value2...`, e.g.: `loss=bpr-max,layers=100,constrained_embedding=True`. Boolean training parameters should be either True or False; parameters that can take a list should use / as the separator (e.g. layers=200/200). Mutually exclusive with the -pf (--parameter_file) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-pf', '--parameter_file', metavar='PARAM_PATH', type=str, help='Alternatively, training parameters can be set using a config file specified in this argument. The config file must contain a single OrderedDict named `gru4rec_params`. The parameters must have the appropriate type (e.g. layers = [100]). Mutually exclusive with the -ps (--parameter_string) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-l', '--load_model', action='store_true', help='Load an already trained model instead of training a model. Mutually exclusive with the -ps (--parameter_string) and the -pf (--parameter_file) arguments and one of the three must be provided.')
parser.add_argument('-s', '--save_model', metavar='MODEL_PATH', type=str, help='Save the trained model to the MODEL_PATH. (Default: don\'t save model)')
parser.add_argument('-t', '--test', metavar='TEST_PATH', type=str, nargs='+', help='Path to the test data set(s) located at TEST_PATH. Multiple test sets can be provided (separate with spaces). (Default: don\'t evaluate the model)')
parser.add_argument('-m', '--measure', metavar='AT', type=int, nargs='+', default=[20], help='Measure recall & MRR at the defined recommendation list length(s). Multiple values can be provided. (Default: 20)')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median', 'tiebreaking'], default='standard', help='Sets how to handle if multiple items in the ranked list have the same prediction score (which is usually due to saturation or an error). See the documentation of evaluate_gpu() in evaluation.py for further details. (Default: standard)')
parser.add_argument('-ss', '--sample_store_size', metavar='SS', type=int, default=10000000, help='GRU4Rec uses a buffer for negative samples during training to maximize GPU utilization. This parameter sets the buffer length. Lower values require more frequent recomputation, higher values use more (GPU) memory. Unless you know what you are doing, you shouldn\'t mess with this parameter. (Default: 10000000)')
parser.add_argument('--sample_store_on_cpu', action='store_true', help='If provided, the sample store will be stored in the RAM instead of the GPU memory. This is not advised in most cases, because it significantly lowers the GPU utilization. This option is provided if for some reason you want to train the model on the CPU (NOT advised).')
parser.add_argument('--test_against_items', metavar='N_TEST_ITEMS', type=int, help='It is NOT advised to evaluate recommender algorithms by ranking a single positive item against a set of sampled negatives. It overestimates recommendation performance and also skewes comparisons, as it affects algorithms differently (and if a different sequence of random samples is used, the results are downright uncomparable). If testing takes too much time, it is advised to sample test sessions to create a smaller test set. However, if the number of items is very high (i.e. ABOVE FEW MILLIONS), it might be impossible to evaluate the model within a reasonable time, even on a smaller (but still representative) test set. In this case, and this case only, one can sample items to evaluate against. This option allows to rank the positive item against the N_TEST_ITEMS most popular items. This has a lesser effect on comparison and it is a much stronger criteria than ranking against randomly sampled items. Keep in mind that the real performcance of the algorithm will still be overestimated by the results, but comparison will be mostly fair. If used, you should NEVER SET THIS PARAMETER BELOW 50000 and try to set it as high as possible (for your required evaluation time). (Default: all items are used as negatives for evaluation)')
args = parser.parse_args()

import os.path
orig_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import datetime as dt
import sys
import time
from collections import OrderedDict
from gru4rec import GRU4Rec
import evaluation
import importlib.util
os.chdir(orig_cwd)

def load_data(fname, gru):
    if fname.endswith('.pickle'):
        print('Loading data from pickle file: {}'.format(fname))
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if gru.session_key not in data.columns:
            print('ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(gru.session_key, fname))
            print('The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.')
            sys.exit(1)
        if gru.item_key not in data.columns:
            print('ERROR. The column specified for item IDs "{}" is not in the data file ({})'.format(gru.item_key, fname))
            print('The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.')
            sys.exit(1)
        if gru.time_key not in data.columns:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(gru.time_key, fname))
            print('The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.')
            sys.exit(1)
    else:
        with open(fname, 'rt') as f:
            header = f.readline().strip().split('\t')
        if gru.session_key not in header:
            print('ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(gru.session_key, fname))
            print('The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.')
            sys.exit(1)
        if gru.item_key not in header:
            print('ERROR. The colmn specified for item IDs "{}" is not in the data file ({})'.format(gru.item_key, fname))
            print('The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.')
            sys.exit(1)
        if gru.time_key not in header:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(gru.time_key, fname))
            print('The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.')
            sys.exit(1)
        print('Loading data from TAB separated file: {}'.format(fname))
        data = pd.read_csv(fname, sep='\t', usecols=[gru.session_key, gru.item_key, gru.time_key], dtype={gru.session_key:'int32', gru.item_key:np.str})
    return data

if (args.parameter_string is not None) + (args.parameter_file is not None) + (args.load_model) != 1:
    print('ERROR. Exactly one of the following parameters must be provided: --parameter_string, --parameter_file, --load_model')
    sys.exit(1)

if args.load_model:
    print('Loading trained model from file: {}'.format(args.path))
    gru = GRU4Rec.loadmodel(args.path)
else:
    if args.parameter_file:
        param_file_path = os.path.abspath(args.parameter_file)
        param_dir, param_file = os.path.split(param_file_path)
        spec = importlib.util.spec_from_file_location(param_file.split('.py')[0], os.path.abspath(args.parameter_file))
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        gru4rec_params = params.gru4rec_params
        print('Loaded parameters from file: {}'.format(param_file_path))
    if args.parameter_string:
        gru4rec_params = OrderedDict([x.split('=') for x in args.parameter_string.split(',')])
    gru = GRU4Rec()
    gru.set_params(**gru4rec_params)
    print('Loading training data...')
    data = load_data(args.path, gru)
    store_type = 'cpu' if args.sample_store_on_cpu else 'gpu'
    if store_type == 'cpu':
        print('WARNING! The sample store is set to be on the CPU. This will make training significantly slower on the GPU.')
    print('Started training')
    t0 = time.time()
    gru.fit(data, sample_store=args.sample_store_size, store_type='gpu')
    t1 = time.time()
    print('Total training time: {:.2f}s'.format(t1 - t0))
    if args.save_model is not None:
        print('Saving trained model to: {}'.format(args.save_model))
        gru.savemodel(args.save_model)

items = None
if args.test_against_items is not None:
    if args.test_against_items < 50000:
        print('ERROR. You musn\'t evaluate positive items agains less than 50000 items.')
        sys.exit(1)
    print('WARNING! You set the number of negative test items. You musn\'t evaluate positive items against a subset of all items unless the number of items in your data is too high (i.e. above a few millions) and evaluation takes too much time.')
    supp = data.groupby('ItemId').size()
    supp.sort_values(inplace=True, ascending=False)
    items = supp[:args.test_against_items].index
    
if args.test is not None:
    for test_file in args.test:
        print('Loading test data...')
        test_data = load_data(test_file, gru)
        for c in args.measure:
            print('Starting evaluation (cut-off={}, using {} mode for tiebreaking)'.format(c, args.eval_type))
            t0 = time.time()
            res = evaluation.evaluate_gpu(gru, test_data, items, batch_size=100, cut_off=c, mode=args.eval_type)
            t1 = time.time()
            print('Evaluation took {:.2f}s'.format(t1 - t0))
            print('Recall@{}: {:.6f} MRR@{}: {:.6f}'.format(c, res[0], c, res[1]))
