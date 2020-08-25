import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    # Datasets 
    parser.add_argument(
        '--video_dir',
        default='dataset/HMDB51/',
        type=str,
        help='path of jpg files')
    parser.add_argument(
        '--annotation_path',
        default='dataset/HMDB51_labels',
        type=str,
        help='label paths')
    parser.add_argument(
        '--dataset',
        default='HMDB51',
        type=str,
        help='(HMDB51, UCF101, Kinectics)')
    parser.add_argument(
        '--split',
        default=1,
        type=str,
        help='(for HMDB51 and UCF101)')
    parser.add_argument(
        '--modality',
        default='RGB',
        type=str,
        help='(RGB, Flow)')
    parser.add_argument(
        '--input_channels',
        default=3,
        type=int,
        help='(3, 2)')
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument(
        '--n_finetune_classes',
        default=51,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    
    # Model parameters
    parser.add_argument(
        '--model_name',
        default='resnext',
        type=str,
        help='Model base architecture')
    parser.add_argument(
        '--model_depth',
        default=101,
        type=int,
        help='Number of layers in model')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--ft_begin_index',
        default=4,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--batch_size', 
        default=32, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_workers', 
        default=4, 
        type=int, 
        help='Number of workers for dataloader')
    parser.add_argument(
        '--pretrained_path', 
        default='', 
        type=str, 
        help='Pretrained model (.pth) of CNN')
    parser.add_argument(
        '--test_md_path', 
        default='', 
        type=str, 
        help='Pretrained model (.pth) of CNN')
    parser.add_argument(
        '--resume_md_path', 
        default='', 
        type=str, 
        help='Pretrained model (.pth) of CNN')

    # optimizer parameters
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Initial learning rate of cnn (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--momentum', 
        default=0.9, 
        type=float, 
        help='Momentum')
    parser.add_argument(
        '--dampening', 
        default=0.9, 
        type=float, 
        help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', 
        default=1e-5, 
        type=float, 
        help='Weight Decay')
    parser.add_argument(
        '--nesterov', 
        action='store_true', 
        help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--n_epochs',
        default=400,
        type=int,
        help='Number of total epochs to run')

    # options for logging
    parser.add_argument(
        '--result_path',
        default='',
        type=str,
        help='result_path')
    
    parser.add_argument(
        '--log',
        default=1,
        type=int,
        help='Log training and validation')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--random_seed', default=1, type=bool, help='Manually set random seed of sampling validation clip')

    parser.add_argument(
        '--cuda',
        action='store_true',
        help='whether use GPU')
    parser.set_defaults(cuda=False)
    parser.add_argument(
        '--highest_val', default={'name':0}, type=dict, help='store best val')
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        type=str, 
        help='Pretrained model (.pth) of CNN')
    
    args = parser.parse_args()
    

    return args
