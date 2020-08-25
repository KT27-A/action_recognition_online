from __future__ import division
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from PIL import Image, ImageFilter
import pickle
import glob
from decord import VideoReader
from .tensor_process import *
import imutils

def get_test_clip_cv2(opts, video_path):
    """
        Args:
            opts         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : list of all video frames

    """

    clip_b = []
    clip_stamps = []
    i = 0

    cap = cv2.VideoCapture(video_path)


    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < opts.sample_duration: 
        single_clip_stamp = list(range(0, total_frames))
        while len(single_clip_stamp) < opts.sample_duration:
            single_clip_stamp.append(i)
            i += 1
            if i >= total_frames:
                i = 0
        clip_stamps.append(single_clip_stamp)
    else:
        s_stamp = list(range(0, total_frames, opts.sample_duration))[:-1]
        # s_stamp.append(total_frames - opts.sample_duration)
        for f_start in s_stamp:
            clip_stamps.append(list(range(f_start, f_start+opts.sample_duration)))
    
    if opts.modality == 'RGB': 
        for stamps in clip_stamps:
            clip_s = []
            for s in stamps:
                (grabbed, frame) = cap.read()
                if grabbed:
                    frame = imutils.resize(frame, height=opts.sample_size)
                    clip_s.append(frame)
            pro_clip = cv2.dnn.blobFromImages(clip_s, 1.0,
            (opts.sample_size, opts.sample_size), (0, 0, 0),
            swapRB=True, crop=True)
            pro_clip = np.transpose(pro_clip, (1, 0, 2, 3))

            # batch = vr.get_batch(stamps).asnumpy()
            # show_img_numpy(batch[0])
            clip_b.append(pro_clip)
            show_img_numpy(pro_clip[:, 0, :, :])

    return torch.from_numpy(np.array(clip_b, dtype=np.float32))


def get_test_clip(opts, video_path):
    """
        Args:
            opts         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : list of all video frames
        """

    clip = []
    clip_stamps = []
    i = 0

    try: vr = VideoReader(video_path, width=-1, height=-1)
    except: 
        print('video path {} cannot be opened'.format(video_path))
        with open('un_opened_file.txt', 'a') as f:
            f.write(video_path)
            f.write('\n')
    
    # h, w = vr[0].shape[:2]
    # if h > w:
    #     r_w = 256
    #     r_h = int(h/w*256)
    # else:
    #     r_h = 256
    #     r_w = int(w/h*256)
    # vr = VideoReader(video_path, width=r_w, height=r_h)


    total_frames = len(vr)
    # in case video FPS >> 30 
    if total_frames > 300:
        s_stamp = np.linspace(0, total_frames, int(300/16)+1)
        s_stamp = s_stamp.astype(np.int)
        for i in range(len(s_stamp[:-1])):
            i_batch = list(np.linspace(s_stamp[i], s_stamp[i+1]-1, 16).astype(np.int))
            clip_stamps.append(i_batch)
    else:
        if total_frames < opts.sample_duration: 
            single_clip_stamp = list(range(0, total_frames))
            while len(single_clip_stamp) < opts.sample_duration:
                single_clip_stamp.append(i)
                i += 1
                if i >= total_frames-1:
                    i = 0
            clip_stamps.append(single_clip_stamp)
        else:
            s_stamp = list(range(0, total_frames, opts.sample_duration))[:-1]
            s_stamp.append(total_frames - opts.sample_duration)
            for f_start in s_stamp:
                clip_stamps.append(list(range(f_start, f_start+opts.sample_duration)))
        
    if opts.modality == 'RGB': 
        for stamps in clip_stamps:
            # batch = vr.get_batch(stamps).asnumpy()
            # show_img_numpy(batch[0])
            clip.append(vr.get_batch(stamps).asnumpy())

    return torch.from_numpy(np.array(clip, dtype=np.float32).transpose(0, 4, 1, 2, 3))


def get_train_clip(opts, video_path):
    """
        Chooses a random clip from a video for training/ validation
        Args:
            opts         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : random clip (list of frames of length sample_duration) from a video for training/ validation
        """
    clip = []
    i = 0
    loop = False

    vr = VideoReader(video_path, width=-1, height=-1)
    # h, w = vr[0].shape[:2]
    # if h > w:
    #     r_w = 256
    #     r_h = int(h/w*256)
    # else:
    #     r_h = 256
    #     r_w = int(w/h*256)
    # vr = VideoReader(video_path, width=r_w, height=r_h)

    total_frames = len(vr)

    if total_frames > 300:
        interval = int(total_frames / (300 / opts.sample_duration))
        s_frame = np.random.randint(0, total_frames - interval)
        f_stamp = list(np.linspace(s_frame, s_frame+interval, 
                    opts.sample_duration).astype(np.int))
        clip = vr.get_batch(f_stamp).asnumpy()
        return torch.from_numpy(clip.transpose(3, 0, 1, 2).astype(np.float32))

    else:
        # choosing a random frame
        if total_frames <= opts.sample_duration: 
            loop = True
            start_frame = 0
        else:
            start_frame = np.random.randint(0, total_frames - opts.sample_duration)
        

        if opts.modality == 'RGB': 
            while len(clip) < opts.sample_duration:
                clip.append(vr.get_batch([start_frame+i]).asnumpy()[0]) # revised
                i += 1
                
                if loop and i == total_frames:
                    i = 0
        
        return torch.from_numpy(np.array(clip, dtype=np.float32).transpose(3, 0, 1, 2))


class HMDB51(Dataset):
    """HMDB51 Dataset"""
    def __init__(self, data_type, opts, split=None):
        """
        Args:
            opts   : config options
            data_type : train for training, val for validation, test for testing 
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        self.lab_names = sorted(set(['_'.join(os.path.splitext(file)[0].split('_')[:-2]) 
                                for file in os.listdir(opts.annotation_path)]))
        
        # Set data augmentation
        if data_type == 'train':
            self.sp_transform = Compose([
                TensorMultiScaleCornerCrop(opts.sample_size),
                TensorFlip(),
                Normalize(get_mean('activitynet'), [1,1,1])
            ])
        else:
            self.sp_transform = Compose([
                TensorScale(opts.sample_size),
                TensorCenterCrop(opts.sample_size),
                Normalize(get_mean('activitynet'), [1,1,1])
            ])
            
        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 51

        self.lab_names = dict(zip(self.lab_names, range(self.N)))   # Each label is mappped to a number

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opts.annotation_path) 
                        if file.strip('.txt')[-1] ==str(split)])
       
        self.data = []  # (filename , lab_id)
        for file in split_lab_filenames:
            class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
            with open(os.path.join(opts.annotation_path, file), 'r') as f:
                for line in f: 
                    # If training data
                    if data_type == 'train' and line.split(' ')[1] == '1':
                        video_path = os.path.join(opts.video_dir, class_id, line.split(' ')[0])
                        if os.path.exists(video_path):
                            self.data.append((line.split(' ')[0], class_id))
                        else:
                            print('Video path {} not exists'.format(video_path))
                        
                    # Elif validation/test data        
                    elif data_type != 'train' and line.split(' ')[1] == '2':
                        # TODO simplify
                        video_path = os.path.join(opts.video_dir, class_id, line.split(' ')[0])
                        if os.path.exists(video_path):
                            self.data.append((line.split(' ')[0], class_id))
                        else:
                            print('Video path {} not exists'.format(video_path))

    def __len__(self):
        '''
        returns number of test/train set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = self.lab_names.get(video[1])
        video_path = os.path.join(self.opts.video_dir, video[1], video[0])
        
        if self.data_type == 'test': 
            clip = get_test_clip(self.opts, video_path)
        else:
            clip = get_train_clip(self.opts, video_path)

        self.sp_transform.randomize_parameters()
        clip = self.sp_transform(clip)

        return clip, label_id
            

class UCF101(Dataset):
    """UCF101 Dataset"""
    def __init__(self, data_type, opts, split=None):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing 
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.data_type = data_type
        self.opts = opts
        if data_type == 'train':
            self.sp_transform = Compose([
                TensorMultiScaleCornerCrop(opts.sample_size),
                TensorFlip(),
                Normalize(get_mean('activitynet'), [1,1,1])
            ])
        else:
            self.sp_transform = Compose([
                TensorScale(opts.sample_size),
                TensorCenterCrop(opts.sample_size), 
                Normalize(get_mean('activitynet'), [1,1,1])
            ])
        
        with open(os.path.join(self.opts.annotation_path, "classInd.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]
        
        with open(os.path.join(self.opts.annotation_path, "classInd.txt")) as lab_file:
            index = [int(line.strip('\n').split(' ')[0]) for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 101

        self.class_idx = dict(zip(self.lab_names, index))   # Each label is mappped to a number
        self.idx_class = dict(zip(index, self.lab_names))   # Each number is mappped to a label

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opts.annotation_path) if file.strip('.txt')[-1] ==str(split)])

        if self.data_type == 'train':
            split_lab_filenames = [f for f in split_lab_filenames if 'train' in f]
        else:
            split_lab_filenames = [f for f in split_lab_filenames if 'test' in f]
        
        self.data = [] # (filename, id)
        
        f = open(os.path.join(self.opts.annotation_path, split_lab_filenames[0]), 'r')
        for line in f:
            class_id = self.class_idx.get(line.split('/')[0]) - 1
            if os.path.exists(os.path.join(self.opts.video_dir, line.split('.')[0]+'.avi')) == True:
                self.data.append((os.path.join(self.opts.video_dir, line.split('.')[0]+'.avi'), class_id))
        f.close()
    def __len__(self):
        '''
        returns number of test set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        video_path = video[0]
        
        if self.data_type == 'test': 
            clip = get_test_clip(self.opts, video_path)
            # clip = get_test_clip_cv2(self.opts, video_path)
        else:
            clip = get_train_clip(self.opts, video_path)

        self.sp_transform.randomize_parameters()
        clip = self.sp_transform(clip)

        return clip, label_id


class SSthv1(Dataset):
    def __init__(self, data_type, opts, sp_transform=None, split=None):
        """
        Args:
            opts   : config options
            data_type : 1 for training, 2 for validation 
            split : 'val' or 'train'
        Returns:
            (tensor(frames), class_id ) : Shape of tensor C x T x H x W
        """
        self.split = split
        self.opts = opts
        self.data_type = data_type
        self.sp_transform = sp_transform

        # joing labnames with underscores
        with open(os.path.join(self.opts.annotation_path, "category.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 174
        
        # indexes for validation set
        if self.data_type == 'train':
            label_file = os.path.join(self.opts.annotation_path, 'train_videofolder.txt')
        else:
            label_file = os.path.join(self.opts.annotation_path, 'val_videofolder.txt')

        self.data = []                                     # (filename , lab_id)
    
        f = open(label_file, 'r')
        for line in f:
            class_id = int(line.strip('\n').split(' ')[-1])
            self.data.append((os.path.join(self.opts.frame_dir,line.strip('\n').split(' ')[0]), class_id))
        f.close()
            
    def __len__(self):
        '''
        returns number of test set
        '''          
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        frame_path = video[0]
        
        if self.opts.modality == 'RGB':
            Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))  
        else:
            # Total_frames = len(glob.glob(glob.escape(frame_path) +  '/TVL1jpg_y_*.jpg'))
            Total_frames = len(glob.glob(glob.escape(frame_path) +  'frame*.jpg'))

        if self.data_type == 'test': 
            clip = get_test_video(self.opts, frame_path, Total_frames)
        else:
            clip = get_train_video(self.opts, frame_path, Total_frames)

        or_clip, ct_clip = clip_to_or_re_tensor(clip, self.opts, self.sp_transform)
        return scale_crop(clip, self.data_type, self.opts), or_clip, ct_clip, label_id

 
class KIN400(Dataset):
    def __init__(self, data_type, opts, split=None):
        """
        Args:
            opts   : config options
            train : train for training, val for validation, test for testing 
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.split = split
        self.opts = opts
        self.data_type = data_type
              
        # joing labnames with underscores
        self.lab_names = sorted([f for f in os.listdir(
                            os.path.join(self.opts.video_dir, "kinetics_400_train"))])        
        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 400

        self.lab_names = dict(zip(self.lab_names, range(self.N)))   # Each label is mappped to a number

        # Set data augmentation
        if data_type == 'train':

            self.sp_transform = Compose([
                TensorScale(opts.sample_size),
                TensorMultiScaleCornerCrop(opts.sample_size),
                TensorFlip(),
                Normalize(get_mean('activitynet'), [1,1,1])
            ])
        else:
            self.sp_transform = Compose([
                TensorScale(opts.sample_size),
                TensorCenterCrop(opts.sample_size),
                Normalize(get_mean('activitynet'), [1,1,1])
            ])

        #TODO make general
        if self.data_type == 'train':
            label_file = os.path.join(self.opts.annotation_path, 'train_list.txt')
            task_annotation = 'kinetics_400_train'
        elif self.data_type == 'val':
            label_file = os.path.join(self.opts.annotation_path, 'val_list.txt')
            task_annotation = 'kinetics_400_val'
        elif self.data_type == 'test':
            label_file = os.path.join(self.opts.annotation_path, 'val_list.txt')
            task_annotation = 'kinetics_400_val'
        self.data = []    # (filename , lab_id)
    
        with open(label_file, 'r') as f:
            for line in f:
                class_id = line.split('/')[0]
                video_path = os.path.join(self.opts.video_dir, 
                    task_annotation, line.strip('\n'))+'.mp4'
                # try:
                #     vr = VideoReader(video_path, width=-1, height=-1)
                #     self.data.append((video_path, class_id))
                # except:
                #     print('Video path {} not exists'.format(video_path)) 
                if os.path.exists(video_path):
                    self.data.append((video_path, class_id))
                else:
                    print('Video path {} not exists'.format(video_path))
                
                
    def __len__(self):
        '''
        returns number of test set
        '''          
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = self.lab_names.get(video[1])
        video_path = video[0]
        
        if self.data_type == 'test': 
            clip = get_test_clip(self.opts, video_path)
        else:
            clip = get_train_clip(self.opts, video_path)
            self.sp_transform.randomize_parameters()

        clip = self.sp_transform(clip)

        return clip, label_id

    
