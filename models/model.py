from __future__ import division
import torch
from torch import nn
from models import resnext
import pdb

def generate_model(opts):
    assert opts.model_name in ['resnext']
    assert opts.model_depth in [101]

    from models.resnext import get_fine_tuning_parameters
    model = resnext.resnet101(
            num_classes=opts.n_classes,
            shortcut_type=opts.resnet_shortcut,
            cardinality=opts.resnext_cardinality,
            sample_size=opts.sample_size,
            sample_duration=opts.sample_duration,
            input_channels=opts.input_channels)
    

    model = model.cuda()
    model = nn.DataParallel(model)
    
    if opts.pretrained_path:
        print('loading pretrained model {}'.format(opts.pretrained_path))
        pretrain = torch.load(opts.pretrained_path)
        
        assert opts.arch == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'])
        model.module.fc = nn.Linear(model.module.fc.in_features, opts.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opts.ft_begin_index)
        return model, parameters

    return model, model.parameters()

