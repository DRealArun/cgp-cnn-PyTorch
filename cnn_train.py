#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import random
from skimage.measure import compare_psnr
from sklearn.metrics import confusion_matrix
import os
import sys
from cnn_model import CGP2CNN
from my_data_loader import get_train_valid_loader, get_test_loader, get_train_test_loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

CIFAR_CLASSES = 10
MNIST_CLASSES = 10
FASHION_CLASSES = 10
EMNIST_CLASSES = 47
SVHN_CLASSES = 10
STL10_CLASSES = 10
DEVANAGARI_CLASSES = 46 

class_dict = {'cifar10': CIFAR_CLASSES,
              'mnist' : MNIST_CLASSES,
              'emnist': EMNIST_CLASSES,
              'fashion': FASHION_CLASSES,
              'svhn': SVHN_CLASSES,
              'stl10': STL10_CLASSES,
              'devanagari' : DEVANAGARI_CLASSES}

inp_channel_dict = {'cifar10': 3,
                    'mnist' : 1,
                    'emnist': 1,
                    'fashion': 1,
                    'svhn': 3,
                    'stl10': 3,
                    'devanagari' : 1}

img_sizes = {'cifar10': 32,
            'mnist' : 28,
            'emnist': 28,
            'fashion': 28,
            'svhn': 32,
            'stl10': 96,
            'devanagari' : 32}

# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self, dataset_name, validation=True, verbose=True, imgSize=32, batchsize=128, datapath='./', seed=2018):
        # dataset_name: name of data set ('bsds'(color) or 'bsds_gray')
        # validation: [True]  model train/validation mode
        #             [False] model test mode for final evaluation of the evolved model
        #                     (raining data : all training data, test data : all test data)
        # verbose: flag of display
        self.verbose = verbose
        self.imgSize = img_sizes[dataset_name]
        self.validation = validation
        self.batchsize = batchsize
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.seed = seed
        print("Dataset details",self.datapath,self.dataset_name,self.batchsize)
        # load dataset
        # if dataset_name == 'cifar10' or dataset_name == 'mnist':
        #     if dataset_name == 'cifar10':
        self.n_class = class_dict[dataset_name]
        self.channel = inp_channel_dict[dataset_name]
        if self.validation:
            self.dataloader, self.test_dataloader = get_train_valid_loader(data_dir=self.datapath, batch_size=self.batchsize, augment=True, random_seed=self.seed, num_workers=1, pin_memory=True, dataset=dataset_name)
            # self.dataloader, self.test_dataloader = loaders[0], loaders[1]
        else:
            # train_dataset = dset.CIFAR10(root='./', train=True, download=True,
            #         transform=transforms.Compose([
            #             transforms.RandomHorizontalFlip(),
            #             transforms.Scale(self.imgSize),
            #             transforms.ToTensor(),
            #             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #         ]))
            # test_dataset = dset.CIFAR10(root='./', train=False, download=True,
            #         transform=transforms.Compose([
            #             transforms.Scale(self.imgSize),
            #             transforms.ToTensor(),
            #             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #         ]))
            # self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(2))
            # self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize, shuffle=True, num_workers=int(2))
            self.dataloader, self.test_dataloader = get_train_test_loader(data_dir=self.datapath, batch_size=self.batchsize, augment=True, random_seed=self.seed, num_workers=1, pin_memory=True, dataset=dataset_name)
        print('train num    ', len(self.dataloader.dataset))
        print('test num    ', len(self.test_dataloader.dataset), self.validation)
            # print('test num     ', len(self.test_dataloader.dataset))
        # else:
        #     print('\tInvalid input dataset name at CNN_train()')
        #     exit(1)

    def __call__(self, cgp, gpuID, epoch_num=200, out_model='mymodel.model'):
        if self.verbose:
            print('GPUID     :', gpuID)
            print('epoch_num :', epoch_num)
            print('batch_size:', self.batchsize)
        
        # model
        torch.backends.cudnn.benchmark = True
        model = CGP2CNN(cgp, self.channel, self.n_class, self.imgSize)
        init_weights(model, 'kaiming')
        model.cuda(gpuID)
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        criterion.cuda(gpuID)
        optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0.0005)
        input = torch.FloatTensor(self.batchsize, self.channel, self.imgSize, self.imgSize)
        input = input.cuda(gpuID)
        label = torch.LongTensor(self.batchsize)
        label = label.cuda(gpuID)

        # Train loop
        for epoch in range(1, epoch_num+1):
            start_time = time.time()
            if self.verbose:
                print('epoch', epoch)
            train_loss = 0
            total = 0
            correct = 0
            ite = 0
            for module in model.children():
                module.train(True)
            for _, (data, target) in enumerate(self.dataloader):
                if self.dataset_name == 'mnist' or self.dataset_name == 'fashion' or self.dataset_name == 'emnist' or self.dataset_name == 'devanagari':
                    data = data[:,0:1,:,:] # for gray scale images
                data = data.cuda(gpuID)
                target = target.cuda(gpuID)
                input.resize_as_(data).copy_(data)
                input_ = Variable(input)
                label.resize_as_(target).copy_(target)
                label_ = Variable(label)
                optimizer.zero_grad()
                try:
                    output = model(input_, None)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0.
                loss = criterion(output, label_)
                train_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                total += label_.size(0)
                correct += predicted.eq(label_.data).cpu().sum().item()
                ite += 1
            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('Train set : Average Acc : {:.4f}'.format(correct/total))
            print('time ', time.time()-start_time)
            if self.validation:
                #print("Checked validation")
                if epoch == 30:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch == epoch_num:
                    for module in model.children():
                        module.train(False)
                    #print("CAlling validation")
                    t_loss = self.__test_per_std(model, criterion, gpuID, input, label)
                    #print("Tloss",t_loss)
                    sys.stdout.flush()
            else:
                flag = 40
                #if epoch == 5:
                #    for param_group in optimizer.param_groups:
                #        tmp = param_group['lr']
                #    tmp *= 10
                #    for param_group in optimizer.param_groups:
                #        param_group['lr'] = tmp
                if epoch % 10 == 0:
                    for module in model.children():
                        module.train(False)
                    t_loss = self.__test_per_std(model, criterion, gpuID, input, label, True)
                #if epoch == 250:
                #    for param_group in optimizer.param_groups:
                #        tmp = param_group['lr']
                #    tmp *= 0.1
                #    for param_group in optimizer.param_groups:
                #        param_group['lr'] = tmp
                #if epoch == 375:
                #    for param_group in optimizer.param_groups:
                #        tmp = param_group['lr']
                #    tmp *= 0.1
                #    for param_group in optimizer.param_groups:
                #        param_group['lr'] = tmp
        # save the model
        #torch.save(model.state_dict(), './model_%d.pth' % int(gpuID))
        torch.save(model.state_dict(), os.path.join('./',str(self.dataset_name)+"_"+str(self.seed),('model_%d.pth' % int(flag))))
        return t_loss

    # For validation/test
    def __test_per_std(self, model, criterion, gpuID, input, label, verbose=False):
        test_loss = 0
        total = 0
        correct = 0
        ite = 0
        predicted_labels = []
        true_labels = []
        #print("Validation Called", len(self.test_dataloader.dataset))
        for _index , (data, target) in enumerate(self.test_dataloader):
            #print("Inside Validation loop")
            if self.dataset_name == 'mnist' or self.dataset_name == 'fashion' or self.dataset_name == 'emnist' or self.dataset_name == 'devanagari':
                data = data[:,0:1,:,:]
            data = data.cuda(gpuID)
            target = target.cuda(gpuID)
            input.resize_as_(data).copy_(data)
            input_ = Variable(input)
            label.resize_as_(target).copy_(target)
            label_ = Variable(label)
            try:
                start_time = time.time()
                output = model(input_, None)
                if verbose :
                    print("Batch index", _index, "time duration",time.time()-start_time) 
            except:
                #print("Returning 0")
                import traceback
                traceback.print_exc()
                return 0.
            loss = criterion(output, label_)
            test_loss += loss.data[0]
            _, predicted = torch.max(output.data, 1)
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(label_.data.tolist())
            total += label_.size(0)
            correct += predicted.eq(label_.data).cpu().sum().item()
            ite += 1
        
        #print("Reached to the end")
        print('Test set : Average loss: {:.4f}'.format(test_loss))
        print('Test set : (%d/%d)' % (correct, total))
        print('Test set : Average Acc : {:.4f}'.format(correct/total))
        #print("Values", predicted_labels[0:5], true_labels[0:5])
        if verbose:
            print("\nSize of predicted labels and true labels", np.shape(predicted_labels), np.shape(true_labels))
            print("Confusion matrix is :\n")
            cm = confusion_matrix(true_labels, predicted_labels)
            for i in range(np.shape(cm)[0]):
                print(cm[i,:])
        sys.stdout.flush()
        return (correct/total)
