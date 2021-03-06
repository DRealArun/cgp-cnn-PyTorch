#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import pandas as pd

from cgp import *
from cgp_config import *
from cnn_train import CNN_train
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

img_sizes = {'cifar10': 32,
            'mnist' : 28,
            'emnist': 28,
            'fashion': 28,
            'svhn': 32,
            'stl10': 96,
            'devanagari' : 32}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evolving CAE structures')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--lam', '-l', type=int, default=2, help='Num. of offsprings')
    parser.add_argument('--net_info_file', default='network_info.pickle', help='Network information file name')
    parser.add_argument('--log_file', default='log_cgp.txt', help='Log file name')
    parser.add_argument('--mode', '-m', default='evolution', help='Mode (evolution / retrain / reevolution)')
    parser.add_argument('--init', '-i', action='store_true')
    parser.add_argument('--dataset', default='cifar10', help='which dataset ? (cifar10, mnist, fashion, devanagari, stl10, svhn, emnist)')
    parser.add_argument('--datapath', default='./', help='root folder for dataset')
    parser.add_argument('--batchsize',type=int, default=16, help='batch size to use')
    parser.add_argument('--seed',type=int, default=0, help='random seed')
    args = parser.parse_args()
    network_file_path = os.path.join('./'+str(args.dataset)+"_"+str(args.seed), args.net_info_file)
    print("Network Info File", network_file_path)
    np.random.seed(args.seed)
    # --- Optimization of the CNN architecture ---
    if args.mode == 'evolution':
        # Create CGP configuration and save network information
        network_info = CgpInfoConvSet(rows=2, cols=34, level_back=10, min_active_num=1, max_active_num=30)
        with open(network_file_path, mode='wb') as f:
            pickle.dump(network_info, f)
        # Evaluation function for CGP (training CNN and return validation accuracy)
        imgSize = img_sizes[args.dataset]
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset=args.dataset, verbose=True, epoch_num=50, batchsize=args.batchsize, imgSize=imgSize, datapath=args.datapath, seed=args.seed)

        # Execute evolution
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize, init=args.init)
        cgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_file=args.log_file, log_dir='./'+str(args.dataset)+"_"+str(args.seed))

    # --- Retraining evolved architecture ---
    elif args.mode == 'retrain':
        print('Retrain')
        # In the case of existing log_cgp.txt
        # Load CGP configuration
        with open(network_file_path, mode='rb') as f:
            network_info = pickle.load(f)
        # Load network architecture
        cgp = CGP(network_info, None)
        file_path = os.path.join('./'+str(args.dataset)+"_"+str(args.seed), args.log_file)
        data = pd.read_csv(file_path, header=None)  # Load log file
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))  # Read the log at final generation
        print(cgp._log_data(net_info_type='active_only', start_time=0))
        # Retraining the network
        temp = CNN_train(args.dataset, validation=False, verbose=True, imgSize=img_sizes[args.dataset], batchsize=args.batchsize, datapath=args.datapath, seed=args.seed)
        file_path = os.path.join('./'+str(args.dataset)+"_"+str(args.seed), 'retrained_net.model')
        acc = temp(cgp.pop[0].active_net_list(), 0, epoch_num=200, out_model=file_path)
        print(acc)

        # # otherwise (in the case where we do not have a log file.)
        # temp = CNN_train('haze1', validation=False, verbose=True, imgSize=128, batchsize=16)
        # cgp = [['input', 0], ['S_SumConvBlock_64_3', 0], ['S_ConvBlock_64_5', 1], ['S_SumConvBlock_128_1', 2], ['S_SumConvBlock_64_1', 3], ['S_SumConvBlock_64_5', 4], ['S_DeConvBlock_3_3', 5]]
        # acc = temp(cgp, 0, epoch_num=500, out_model='retrained_net.model')

    elif args.mode == 'reevolution':
        # restart evolution
        print('Restart Evolution')
        imgSize = img_sizes[args.dataset]
        with open(network_file_path, mode='rb') as f:
            network_info = pickle.load(f)
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset=args.dataset, verbose=True, epoch_num=50, batchsize=args.batchsize, imgSize=img_sizes[args.dataset])
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=img_sizes[args.dataset])
        file_path = os.path.join('./'+str(args.dataset)+"_"+str(args.seed), args.log_file)
        data = pd.read_csv(file_path, header=None)
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))
        cgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_file=args.log_file, log_dir='./'+str(args.dataset)+"_"+str(args.seed))

    else:
        print('Undefined mode. Please check the "-m evolution or retrain or reevolution" ')
