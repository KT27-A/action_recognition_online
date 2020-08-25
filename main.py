from data_process.datasets import *
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import sys
from utils import *
from train import train_epoch, val_epoch


def main():
    opts = parse_opts()
    if torch.cuda.is_available():
        opts.cuda = True
    opts.device = torch.device(opts.device if opts.cuda else 'cpu')
    print(opts)
    
    opts.arch = '{}-{}'.format(opts.model_name, opts.model_depth)
    torch.manual_seed(opts.manual_seed)

    print("Preprocessing train data ...")
    train_data = globals()['{}'.format(opts.dataset)](data_type='train', opts=opts, split=opts.split)
    print("Length of train data = ", len(train_data))

    print("Preprocessing validation data ...")
    val_data = globals()['{}'.format(opts.dataset)](data_type='val', opts=opts, split=opts.split)
    print("Length of validation data = ", len(val_data))
    
    if opts.modality=='RGB': opts.input_channels = 3
    elif opts.modality=='Flow': opts.input_channels = 2

    print("Preparing dataloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opts.batch_size, 
                                    shuffle=True, num_workers = opts.n_workers, 
                                    pin_memory = True, drop_last=True)
    val_dataloader   = DataLoader(val_data, batch_size = opts.batch_size, 
                                    shuffle=True, num_workers = opts.n_workers, 
                                    pin_memory = True, drop_last=True)
    print("Length of train dataloader = ",len(train_dataloader))
    print("Length of validation dataloader = ",len(val_dataloader))    
   
    # define the model 
    print("Loading model... ", opts.model_name, opts.model_depth)
    model, parameters = generate_model(opts)
    if not opts.pretrained_path:
        opts.learning_rate = 0.1
        opts.weight_decay = 5e-4
    criterion = nn.CrossEntropyLoss().cuda()

    if opts.resume_md_path:
        print('loading checkpoint {}'.format(opts.resume_path1))
        checkpoint = torch.load(opts.resume_path1)
        
        assert opts.arch == checkpoint['arch']
        opts.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    
    log_path = os.path.join(opts.result_path, opts.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    if opts.log == 1:
        if opts.resume_md_path:
            begin_epoch = int(opts.resume_path1.split('/')[-1].split('_')[1])
            train_logger = Logger(os.path.join(log_path, '{}_train_clip{}model{}{}.log'
                        .format(opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth))
                        ,['epoch', 'loss', 'acc', 'lr'], overlay=False)
            val_logger   = Logger(os.path.join(log_path, '{}_val_clip{}model{}{}.log'
                            .format(opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth))
                            ,['epoch', 'loss', 'acc'], overlay=False)
        else:
            begin_epoch = 0
            train_logger = Logger(os.path.join(log_path, '{}_train_clip{}model{}{}.log'
                        .format(opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth))
                        ,['epoch', 'loss', 'acc', 'lr'], overlay=True)
            val_logger   = Logger(os.path.join(log_path, '{}_val_clip{}model{}{}.log'
                            .format(opts.dataset, opts.sample_duration, opts.model_name, opts.model_depth))
                            ,['epoch', 'loss', 'acc'], overlay=True)
            
           
    print("Initializing the optimizer ...")

    if opts.nesterov: dampening = 0
    else: dampening = opts.dampening
        
    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opts.learning_rate, opts.momentum, dampening, opts. weight_decay, opts.nesterov))
    print("LR patience = ", opts.lr_patience)
    
    
    optimizer = optim.SGD(
        parameters,
        lr=opts.learning_rate,
        momentum=opts.momentum,
        dampening=dampening,
        weight_decay=opts.weight_decay,
        nesterov=opts.nesterov)


    if opts.resume_md_path:
        optimizer.load_state_dict(torch.load(opts.pretrained_path)['optimizer'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opts.lr_patience)

    for epoch in range(begin_epoch, opts.n_epochs + 1):
        print('Start training epoch {}'.format(epoch))
        train_epoch(epoch, train_dataloader, model, criterion, optimizer, opts,
                        train_logger)

        print('Start validating epoch {}'.format(epoch))
        with torch.no_grad():
            val_epoch(epoch, val_dataloader, model, criterion, optimizer, opts,
                    val_logger, scheduler)


if __name__=="__main__":
    main()
        



