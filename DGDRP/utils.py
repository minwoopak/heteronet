import hashlib
import json
import os
from os import listdir
from os.path import isfile, join
import pandas as pd


def save_exp_result(setting, result, dir_path):
    ''' Save result dictionaries as JSON file'''
    exp_name = setting['exp_name']
    #del setting['max_epoch']
    #del setting['train_batch_size']
    #del setting['test_batch_size']

    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = dir_path+'/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)
    with open(filename, 'w') as f:
        json.dump(result, f)


def load_exp_result(exp_name, dir_path):
    ''' Load JSON file and convert to df '''
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) if '.json' in f]
    list_result = []
    for filename in filenames:
        if exp_name in filename:
            with open(join(dir_path, filename), 'r') as infile:
                results = json.load(infile)
                list_result.append(results)
    df = pd.DataFrame(list_result) # .drop(columns=[])
    return df


def logging(msg, outdir, log_fpath):
    fpath = os.path.join(outdir, log_fpath)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    with open(fpath, 'a') as fw:
        fw.write("%s\n" % msg)
    print(msg)
