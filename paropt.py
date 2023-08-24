import argparse
import os
import optuna
import json

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
parser.add_argument('test', metavar='TEST_PATH', type=str, help='Path to the test data set(s) located at TEST_PATH.')
parser.add_argument('-g', '--gru4rec_model', metavar='GRFILE', type=str, default='gru4rec', help='Name of the file containing the GRU4Rec class. Can be sued to select different varaiants. (Default: gru4rec)')
parser.add_argument('-tf', '--theano_flags', metavar='FLAGS', type=str, nargs='?', default='device=cuda0', help='Theano settings.')
parser.add_argument('-fp', '--fixed_parameters', metavar='PARAM_STRING', type=str, help='Fixed training parameters provided as a single parameter string. The format of the string is `param_name1=param_value1,param_name2=param_value2...`, e.g.: `loss=bpr-max,layers=100,constrained_embedding=True`. Boolean training parameters should be either True or False; parameters that can take a list should use / as the separator (e.g. layers=200/200). Mutually exclusive with the -pf (--parameter_file) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-opf', '--optuna_parameter_file', metavar='PATH', type=str, help='File describing the parameter space for optuna.')
parser.add_argument('-m', '--measure', metavar='AT', type=int, nargs='?', default=20, help='Measure recall & MRR at the defined recommendation list length. A single values can be provided. (Default: 20)')
parser.add_argument('-nt', '--ntrials', metavar='NT', type=int, nargs='?', default=50, help='Number of optimization trials to perform (Default: 50)')
parser.add_argument('-fm', '--final_measure', metavar='AT', type=int, nargs='*', default=[20], help='Measure recall & MRR at the defined recommendation list length(s) after the optimization is finished. Multiple values can be provided. (Default: 20)')
parser.add_argument('-pm', '--primary_metric', metavar='METRIC', choices=['recall', 'mrr'], default='recall', help='Set primary metric, recall or mrr (e.g. for paropt). (Default: recall)')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median', 'tiebreaking'], default='standard', help='Sets how to handle if multiple items in the ranked list have the same prediction score (which is usually due to saturation or an error). See the documentation of evaluate_gpu() in evaluation.py for further details. (Default: standard)')
parser.add_argument('-ik', '--item_key', metavar='IK', type=str, default='ItemId', help='Column name corresponding to the item IDs (detault: ItemId).')
parser.add_argument('-sk', '--session_key', metavar='SK', type=str, default='SessionId', help='Column name corresponding to the session IDs (default: SessionId).')
parser.add_argument('-tk', '--time_key', metavar='TK', type=str, default='Time', help='Column name corresponding to the timestamp (default: Time).')

args = parser.parse_args()

import pexpect
import numpy as np
from collections import OrderedDict
import importlib
import re

def generate_command(optimized_param_str):
    command = 'python run.py "{}" -t "{}" -g {} -ps {},{} -m {} -pm {} -lpm -e {} -ik {} -sk {} -tk {}'.format(args.path, args.test, args.gru4rec_model, args.fixed_parameters, optimized_param_str, args.measure, args.primary_metric, args.eval_type, args.item_key, args.session_key, args.time_key)
    return command

def run_once(optimized_param_str):
    command = generate_command(optimized_param_str)
    os.environ['THEANO_FLAGS'] = args.theano_flags
    cmd = pexpect.spawnu(command, timeout=None, maxread=1)
    line = cmd.readline()
    while line:
        line = line.strip()
        print(line)
        if re.match('PRIMARY METRIC: -*\\d\\.\\d+e*-*\\d*', line):
            t = line.split(':')[1].lstrip()
            val = float(t)
            break
        line = cmd.readline()
    return val

class Parameter:
    def __init__(self, name, dtype, values, step=None, log=False):
        assert dtype in ['int', 'float', 'categorical']
        assert type(values)==list
        assert len(values)==2 or dtype=='categorical'
        self.name = name
        self.dtype = dtype
        self.values = values
        self.step = step
        if self.step is None and self.dtype=='int':
            self.step = 1
        self.log = log
    @classmethod
    def fromjson(cls, json_string):
        obj = json.loads(json_string)
        return Parameter(obj['name'], obj['dtype'], obj['values'], obj['step'] if 'step' in obj else None, obj['log'] if 'log' in obj else False)
    def __call__(self, trial):
        if self.dtype == 'int':
            return trial.suggest_int(self.name, int(self.values[0]), int(self.values[1]), step=self.step, log=self.log)
        if self.dtype == 'float':
            return trial.suggest_float(self.name, float(self.values[0]), float(self.values[1]), step=self.step, log=self.log)
        if self.dtype == 'categorical':
            return trial.suggest_categorical(self.name, self.values)
    def __str__(self):
        desc = 'PARAMETER {} \t type={}'.format(self.name, self.dtype)
        if self.dtype == 'int' or self.dtype == 'float':
            desc += ' \t range=[{}..{}] (step={}) \t {} scale'.format(self.values[0], self.values[1], self.step if self.step is not None else 'N/A', 'UNIFORM' if not self.log else 'LOG')
        if self.dtype == 'categorical':
            desc += ' \t options: [{}]'.format(','.join([str(x) for x in self.values]))
        return desc
        
def objective(trial, par_space):
    optimized_param_str = []
    for par in par_space:
        val = par(trial)
        optimized_param_str.append('{}={}'.format(par.name,val))
    optimized_param_str = ','.join(optimized_param_str)
    val = run_once(optimized_param_str)
    return val

par_space = []
with open(args.optuna_parameter_file, 'rt') as f:
    print('-'*80)
    print('PARAMETER SPACE')
    for line in f:
        par = Parameter.fromjson(line)
        print('\t' + str(par))
        par_space.append(par)
    print('-'*80)

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, par_space), n_trials=args.ntrials)

print('Running final eval @{}:'.format(args.final_measure))
optimized_param_str = ','.join(['{}={}'.format(k,v) for k,v in study.best_params.items()])
os.environ['THEANO_FLAGS'] = args.theano_flags
command = 'python run.py "{}" -t "{}" -g {} -ps {},{} -m {} -e {} -ik {} -sk {} -tk {}'.format(args.path, args.test, args.gru4rec_model, args.fixed_parameters, optimized_param_str, ' '.join([str(x) for x in args.final_measure]), args.eval_type, args.item_key, args.session_key, args.time_key)
cmd = pexpect.spawnu(command, timeout=None, maxread=1)
line = cmd.readline()
while line:
    line = line.strip()
    print(line)
    line = cmd.readline()
