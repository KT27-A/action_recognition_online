from PIL import Image, ImageFilter, ImageOps, ImageChops
import numpy as np
#from torchvision.transforms import *
import torch
import random
import numbers
import time
import cv2
import torch.nn.functional as F

try:
    import accimage
except ImportError:
    accimage = None
    
scale_choice = [1, 1/2**0.25, 1/2**0.5, 1/2**0.75, 0.5]
crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)
            
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'RGB':
            # img = torch.from_numpy(np.array(pic, np.float32, copy=False).transpose(2, 0, 1))
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size
        Returns:
            Tensor: Normalized image.
        """

        tensor_normal = tensor.clone()
        if tensor.dim() == 4:
            for i in range(tensor.shape[0]):
                tensor_normal[i, :, :, :] = (tensor[i, :, :, :] - self.mean[i])/self.std[i]
        elif tensor.dim() == 5:
            for i in range(tensor.shape[1]):
                tensor_normal[:, i, :, :, :] = (tensor[:, i, :, :, :] - self.mean[i])/self.std[i]
        else:
            raise Exception("Normalization dimension is unsupported")  

        return tensor_normal

    def randomize_parameters(self):
        pass

        
def get_mean( dataset='HMDB51'):
    #assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [114.7748, 107.7354, 99.4750 ]
    elif dataset == 'kinetics':
    # Kinetics (10 videos for each class)
        return [110.63666788, 103.16065604,  96.29023126]
    elif dataset == "HMDB51":
        return [0.36410178082273*255, 0.36032826208483*255, 0.31140866484224*255]

def get_std(dataset = 'HMDB51'):
# Kinetics (10 videos for each class)
    if dataset == 'kinetics':
        return [38.7568578, 37.88248729, 40.02898126]
    elif dataset == 'HMDB51':
        return [0.20658244577568*255, 0.20174469333003*255, 0.19790770088352*255]


class Nor_recover(object):
    """Recover an image to numpy with mean and standard deviation from normalized tensor.
    Given mean: (R, G, B) and std: (R, G, B),
    channel = channel * std + mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor.numpy()


def show_img_numpy(img):
    if isinstance(img, np.ndarray):
        img = img.astype(np.uint8).transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    elif isinstance(img, torch.Tensor):
        img = img.numpy()
        img = img.astype(np.uint8).transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        cv2.waitKey(0)


def show_images(clip, actions, sample_size=112):
    img_bag = []
    actions_np = actions.cpu().numpy().astype(np.int16)
    clip = clip.cpu()
    if clip.shape[0] >= 3:
        show_clip_num = 3
    else:
        show_clip_num = 1
    for i in range(show_clip_num):
        clip_item = clip[i].clone()
        for j in range(3):
            img_nor = clip_item[:, j*7, :, :]
            img = Nor_recover(get_mean('activitynet'), [1,1,1])(img_nor)
            img = img.astype(np.uint8).transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if actions.shape[1] == 2:
                x, y = actions_np[i][0], actions_np[i][1]
                cv2.rectangle(img, (x, y), (x+112, y+112), (0, 0, 255), 2)
                text = '{}'.format((x, y))
            elif actions.shape[1] == 3:
                x, y, s = actions_np[i][0], actions_np[i][1], actions_np[i][2]
                cv2.rectangle(img, (x, y), (min(x + s, 341), min(y + s, 256)), (0, 0, 255), 2)
                text = '{}'.format((x, y, s, s))
            elif actions.shape[1] == 4:
                x, y, w, h = actions_np[i][0], actions_np[i][1], actions_np[i][2], actions_np[i][3]
                cv2.rectangle(img, (x, y), (min(x + w, 341), min(y + h, 256)), (0, 0, 255), 2)
                text = '{}'.format((x, y, w, h))
            cv2.putText(img, text, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
            img_bag.append(img)
            j = 0
    horizon = []
    for i in range(0, len(img_bag), 3):
        horizon.append(np.hstack((img_bag[i], img_bag[i+1], img_bag[i+2])))
    vertical = horizon[0]
    for i in range(1, len(horizon)):
        vertical = np.vstack((vertical, horizon[i])) 
    cv2.imshow('Trained actions', vertical)
    cv2.waitKey(10)


def tensor_resize(clip, sample_size, interpolation='bilinear'):
    if isinstance(sample_size, int):
        desired_size = (sample_size, sample_size)
    elif isinstance(sample_size, tuple):
        desired_size = sample_size
        
    if clip.dim() == 3:
        clip = clip.unsqueeze(0)
        clip_resize = F.interpolate(clip, desired_size, mode=interpolation, align_corners=True)
    elif clip.dim() == 4:
        clip_resize = F.interpolate(clip, desired_size, mode=interpolation, align_corners=True)
    elif clip.dim() == 5:
        clip_resize = torch.zeros((clip.shape[0], clip.shape[1], clip.shape[2], desired_size[0], desired_size[1]))
        for i, clip_item in enumerate(clip):
            clip_resize[i] = F.interpolate(clip_item, desired_size, mode=interpolation, align_corners=True)
            # show_img_numpy(clip_resize[0, :, 0, :, :])

    return clip_resize
     

class TensorResize(object):
    """Resize a tensor as settings.
    Args:
        x (tensor): Sequence of means for R, G, B channels respecitvely.
        sample_size (int): size of samples
    """
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
    
    def __call__(self, x):
        return tensor_resize(x, self.sample_size)

    def randomize_parameters(self):
        pass


class TensorRandomCropSampleSize(object):
    """Randomly crop a tensor with sample size.
    Args:
        x (tensor): Sequence of means for R, G, B channels respecitvely.
        sample_size (int): size of samples
    """
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
    
    def __call__(self, x):
        
        h, w = x.shape[-2:]
        s_x = int(self.r_x * (w - self.sample_size))
        s_y = int(self.r_y * (h - self.sample_size))
        if x.dim() == 4:
            x_cr = x[:, :, s_y:s_y+self.sample_size, s_x:s_x+self.sample_size]
        elif x.dim() == 5:
            x_cr = x[:, :, :, s_y:s_y+self.sample_size, s_x:s_x+self.sample_size]
        
        return x_cr

    def randomize_parameters(self):
        self.r_x = random.random()
        self.r_y = random.random()


class TensorRandomCropSmallSide(object): # to be tested
    """Randomly crop a tensor with the size of smaller side of the tensor.
    Args:
        x (tensor): Sequence of means for R, G, B channels respecitvely.
        sample_size (int): size of samples
    """
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
    
    def __call__(self, x):
        
        h, w = x.shape[-2:]
        crop_size = min(h, w)

        s_x = int(self.r_x * (w - crop_size))
        s_y = int(self.r_y * (h - crop_size))
        if x.dim() == 4:
            x_cr = x[:, :, s_y:s_y+crop_size, s_x:s_x+crop_size]
        elif x.dim() == 5:
            x_cr = x[:, :, :, s_y:s_y+crop_size, s_x:s_x+crop_size]
        x_cr = tensor_resize(x_cr, self.sample_size)
        
        return x_cr

    def randomize_parameters(self):
        self.r_x = random.random()
        self.r_y = random.random()


class TensorCenterCrop(object): 
    
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size

    def __call__(self, x):
        h, w = x.shape[-2:]
        s_x = round((w-self.sample_size)/2)
        s_y = round((h-self.sample_size)/2)
        if x.dim() == 4:
            x_cr = x[:, :, s_y:s_y+self.sample_size, s_x:s_x+self.sample_size].clone()
        elif x.dim() == 5:
            x_cr = x[:, :, :, s_y:s_y+self.sample_size, s_x:s_x+self.sample_size].clone()

        
        return x_cr
        
    def randomize_parameters(self):
        pass


class TensorCornerCrop(object):
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size

    def __call__(self, x):
        h, w = x.shape[-2:]
        if self.pick_loc == 'tl':
            s_x = 0
            s_y = 0
        elif self.pick_loc == 'tr':
            s_x = int(w - self.sample_size)
            s_y = 0
        elif self.pick_loc == 'c':
            s_x = round(w/2) - int(self.sample_size/2)
            s_y = round(h/2) - int(self.sample_size/2)
        elif self.pick_loc == 'bl':
            s_x = 0
            s_y = int(h - self.sample_size)
        elif self.pick_loc == 'br':
            s_x = int(w - self.sample_size)
            s_y = int(h - self.sample_size)
        if x.dim() == 4:
            x_cr = x[:, :, s_y:s_y+self.sample_size, s_x:s_x+self.sample_size]
        elif x.dim() == 5:
            x_cr = x[:, :, :, s_y:s_y+self.sample_size, s_x:s_x+self.sample_size]

        return x_cr

    def randomize_parameters(self):
        self.pick_loc = crop_positions[random.randint(0, len(crop_positions)-1)]
        


class TensorMultiScaleCornerCrop(object):
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size

    def __call__(self, x):
        h, w = x.shape[-2:]
        crop_size = min(h, w)
        crop_size = int(crop_size * self.pick_scale)
        if self.pick_loc == 'tl':
            s_x = 0
            s_y = 0
        elif self.pick_loc == 'tr':
            s_x = int(w - crop_size)
            s_y = 0
        elif self.pick_loc == 'c':
            s_x = round(w/2) - int(crop_size/2)
            s_y = round(h/2) - int(crop_size/2)
        elif self.pick_loc == 'bl':
            s_x = 0
            s_y = int(h - crop_size)
        elif self.pick_loc == 'br':
            s_x = int(w - crop_size)
            s_y = int(h - crop_size)
        if x.dim() == 4:
            x_cr = x[:, :, s_y:s_y+crop_size, s_x:s_x+crop_size]
        elif x.dim() == 5:
            x_cr = x[:, :, :, s_y:s_y+crop_size, s_x:s_x+crop_size]
        x_cr = tensor_resize(x_cr, self.sample_size)
    
        return x_cr
    
    def randomize_parameters(self):
        self.pick_loc = crop_positions[random.randint(0, len(crop_positions)-1)]
        self.pick_scale = scale_choice[random.randint(0, len(scale_choice)-1)]


class TensorThumbNail(object):
    """Randomly flip a tensor.
    Args:
        x (Image): Sequence of means for R, G, B, T (optional) channels respecitvely.
        sample_size (int): size of samples
    """
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size

    def __call__(self, x):
        if x.dim() == 3: # H x W x C
            h, w = x.shape[:2]
            if h > w:
                if (h-w) % 2 == 0:
                    pad_num = int((h-w)/2)
                    pad = (0, 0, pad_num, pad_num, 0, 0)
                else:
                    pad_num = int((h-w)/2)
                    pad = (0, 0, pad_num, pad_num+1, 0, 0)
            else:
                if (w-h) % 2 == 0:
                    pad_num = int((w-h)/2)
                    pad = (0, 0, 0, 0, pad_num, pad_num)
                else:
                    pad_num = int((w-h)/2)
                    pad = (0, 0, 0, 0, pad_num, pad_num+1)
        elif x.dim() == 4: # C x T x H x W
            h, w = x.shape[-2:]
            if h > w:
                if (h-w) % 2 == 0:
                    pad_num = int((h-w)/2)
                    pad = (pad_num, pad_num, 0, 0, 0, 0, 0, 0)
                else:
                    pad_num = int((h-w)/2)
                    pad = (pad_num, pad_num+1, 0, 0, 0, 0, 0, 0)
            else:
                if (w-h) % 2 == 0:
                    pad_num = int((w-h)/2)
                    pad = (0, 0, pad_num, pad_num, 0, 0, 0, 0)
                else:
                    pad_num = int((w-h)/2)
                    pad = (0, 0, pad_num, pad_num+1, 0, 0, 0, 0)
        self.ori_h = h
        self.ori_w = w
        self.pad_num = pad_num
        x_out = F.pad(x, pad, "constant", 0)
        # x_out = tensor_resize(x_out, self.sample_size)
        return x_out
        
    def remove_pad(self, x):
        if x.dim() == 3: # H x W x C
            if self.ori_h > self.ori_w:
                x_out = x[:, self.pad_num:self.pad_num+self.ori_w, :]
            else:
                x_out = x[self.pad_num:self.pad_num+self.ori_h, :, :]
        elif x.dim() == 4: # C x T x H x W
            if self.ori_h > self.ori_w:
                x_out = x[:, :, :, self.pad_num:self.pad_num+self.ori_w]
            else:
                x_out = x[:, :, self.pad_num:self.pad_num+self.ori_h, :]

        return x_out

    def randomize_parameters(self):
        pass


class ImageThumbNail(object): 
    """Randomly flip a tensor.
    Args:
        x (Image): Sequence of means for R, G, B channels respecitvely.
        sample_size (int): size of samples
    """
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size

    def __call__(self, x):
        x_whole = x.copy()
        x_whole.thumbnail((self.sample_size, self.sample_size))
        x_board = Image.new('RGB', (self.sample_size, self.sample_size))
        if self.sample_size > x_whole.size[1]:
            x_board.paste(x_whole, (0, int((self.sample_size-x_whole.size[1])/2)))
        else:
            x_board.paste(x_whole, (int((self.sample_size-x_whole.size[0])/2), 0))
        return x_board

    def randomize_parameters(self):
        pass


class TensorFlip(object):
    """Randomly flip a tensor.
    Args:
        x (tensor): Sequence of means for R, G, B channels respecitvely.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        if self.flip_prob > 0.5:
            return torch.flip(x, [-1])
        else:
            return x
    
    def randomize_parameters(self):
        self.flip_prob = random.random()


class TensorScale(object): # to test
    """Scale images to given size by the shorter side length while keeping the propotion.
    Args:
        x (tensor): Sequence of means for R, G, B channels respecitvely.
        size (int): Desired size of given image 
        interpolation: Desired interpolation method
    """
    def __init__(self, size, interpolation='bilinear'):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def __call__(self, x):
        
        h, w = x.shape[-2:]
        if h > w:
            r_w = self.size
            r_h = int(self.size * (h/w))
        else:
            r_h = self.size
            r_w = int(self.size * (w/h))

        x_scale = tensor_resize(x, (r_h, r_w), interpolation=self.interpolation)
        return x_scale

    
    def randomize_parameters(self):
        pass


def make_clip(clip, opts):
    """make three clips - random cropped clip, thumbtailed clip and centercropped clip
    Args:
        clip (list): a list of Images 
        opts : arguments containing transforms 
        'totensor': totensor_transform,
        'c_crop': center_crop,
        'mc_crop': multicorner_crop,
        'thumbnail':thumbnail
    Returns:
        three clips : Shape of tensor C x T x H x W
    """
    rc_clip = []
    thumb_clip = []
    ct_clip = []
    
    thumb_clip = opts.sp_transform['thumbnail'](clip)
    thumb_clip = opts.sp_transform['norm'](thumb_clip)
    or_clip = opts.sp_transform['thumbnail'].remove_pad(thumb_clip)
    opts.sp_transform['mc_crop'].randomize_parameters()

    rc_clip = opts.sp_transform['mc_crop'](or_clip)
    # ct_clip = opts.sp_transform['c_crop'](or_clip)
    thumb_clip_f = tensor_resize(thumb_clip, 341)
    thumb_clip_c = tensor_resize(thumb_clip, 112)
    
    return rc_clip, thumb_clip_c, thumb_clip_f


  



    
    

        



    

        


    



            
            
            
    
    
    
