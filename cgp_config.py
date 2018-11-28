#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import multiprocessing.pool
import numpy as np
import cnn_train as cnn
import sys

# wrapper function for multiprocessing
def arg_wrapper_mp(args):
    return args[0](*args[1:])

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


# Evaluation of CNNs
def cnn_eval(net, gpu_id, epoch_num, batchsize, dataset, verbose, imgSize, datapath, seed):

    print('\tgpu_id:', gpu_id, ',', net, ',',seed)
    train = cnn.CNN_train(dataset, validation=True, verbose=verbose, imgSize=imgSize, batchsize=batchsize, datapath=datapath, seed=seed)
    evaluation = train(net, gpu_id, epoch_num=epoch_num, out_model=None)
    print('\tgpu_id:', gpu_id, ', eval:', evaluation)
    sys.stdout.flush()
    return evaluation


class CNNEvaluation(object):
    def __init__(self, gpu_num, dataset='cifar10', verbose=True, epoch_num=50, batchsize=16, imgSize=32, datapath='./', seed=0):
        self.gpu_num = gpu_num
        self.epoch_num = epoch_num
        self.batchsize = batchsize
        self.dataset = dataset
        self.verbose = verbose
        self.imgSize = imgSize
        self.datapath = datapath
        self.seed = seed

    def __call__(self, net_lists):
        evaluations = np.zeros(len(net_lists))
        for i in np.arange(0, len(net_lists), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(net_lists))) - i
            pool = NoDaemonProcessPool(process_num)
            arg_data = [(cnn_eval, net_lists[i+j], j, self.epoch_num, self.batchsize, self.dataset, self.verbose, self.imgSize, self.datapath, self.seed) for j in range(process_num)]
            evaluations[i:i+process_num] = pool.map(arg_wrapper_mp, arg_data)
            pool.terminate()

        return evaluations


# network configurations
class CgpInfoConvSet(object):
    def __init__(self, rows=30, cols=40, level_back=40, min_active_num=8, max_active_num=50):
        self.input_num = 1
        # "S_" means that the layer has a convolution layer without downsampling.
        # "D_" means that the layer has a convolution layer with downsampling.
        # "Sum" means that the layer has a skip connection.
        self.func_type = ['S_ConvBlock_32_1',    'S_ConvBlock_32_3',   'S_ConvBlock_32_5',
                          'S_ConvBlock_128_1',    'S_ConvBlock_128_3',   'S_ConvBlock_128_5',
                          'S_ConvBlock_64_1',     'S_ConvBlock_64_3',    'S_ConvBlock_64_5',
                          'S_ResBlock_32_1',     'S_ResBlock_32_3',    'S_ResBlock_32_5',
                          'S_ResBlock_128_1',     'S_ResBlock_128_3',    'S_ResBlock_128_5',
                          'S_ResBlock_64_1',      'S_ResBlock_64_3',     'S_ResBlock_64_5',
                          'Concat', 'Sum',
                          'Max_Pool', 'Avg_Pool']
                          
        self.func_in_num = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            2, 2,
                            1, 1]

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])
