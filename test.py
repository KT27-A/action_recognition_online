from data_process.datasets import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
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
import time
import torch.utils
import sys
from utils import *


if __name__=="__main__":
    # print configuration options
    opts = parse_opts()
    if torch.cuda.is_available():
        opts.cuda = True
    opts.device = torch.device(opts.device if opts.cuda else 'cpu')
    print(opts)
    opts.arch = '{}-{}'.format(opts.model_name, opts.model_depth)

    print("Preprocessing testing data ...")
    test_data = globals()['{}'.format(opts.dataset)](data_type='test', opts=opts, split=opts.split)
    print("Length of testing data = ", len(test_data))
    
    if opts.modality=='RGB': opts.input_channels = 3
    elif opts.modality=='Flow': opts.input_channels = 2

    print("Preparing datatloaders ...")
    test_dataloader = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=opts.n_workers, 
                    pin_memory=True, drop_last=False)
    print("Length of validation datatloader = ",len(test_dataloader))
    
    # Loading model and checkpoint
    model, parameters = generate_model(opts)
    criterion_rl = nn.CrossEntropyLoss(reduction='none').cuda() 
    accuracies = AverageMeter()
    clip_accuracies = AverageMeter()
    
    #Path to store results
    result_path = "{}/{}/".format(opts.result_path, opts.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)    

    if opts.log:
        f = open(os.path.join(result_path, "test_{}{}_{}_{}_{}_{}_plusone.txt".format(opts.model_name, 
            opts.model_depth, opts.dataset, opts.split, opts.modality, opts.sample_duration)), 'w+')
        f.write(str(opts))
        f.write('\n')
        f.flush()
    if opts.test_md_path:
        print('loading checkpoint {}'.format(opts.test_md_path))
        checkpoint = torch.load(opts.test_md_path)
        assert opts.arch == checkpoint['arch']
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():   
        for i, (inputs, targets) in enumerate(test_dataloader):

            inputs = inputs[0]
            inputs = inputs.to(opts.device)
            targets = targets.to(opts.device)
            outputs_var = model(inputs)

            pred5 = np.array(torch.mean(outputs_var, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
               
            acc = float(pred5[0] == targets[0])
                            
            accuracies.update(acc, 1)            
            
            # line = "Video[" + str(i) + "] : \t top5 " + str(pred5) + "\t top1 = " + str(pred5[0]) +  "\t true = " +str(int(targets[0])) + "\t video = " + str(accuracies.avg)
            line = 'Video[{}]:\ttop5 = {}\ttop1 = {}\tgt = {}\tacc = {}'.format(i, pred5, pred5[0], targets[0], accuracies.avg)
            print(line)
            if opts.log:
                f.write(line + '\n')
                f.flush()
    
    print("Video accuracy = ", accuracies.avg)
    line = "Video accuracy = " + str(accuracies.avg) + '\n'
    if opts.log:
        f.write(line)
    
