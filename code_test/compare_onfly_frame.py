from decord import VideoReader
import cv2
from PIL import Image
import time
import numpy as np

if __name__ == "__main__":
    video_path = './v_ApplyEyeMakeup_g01_c01.avi'
    s = time.time()
    vr = VideoReader(video_path, width=341, height=256)
    frame_list = range(16)
    frames = vr.get_batch(frame_list).asnumpy()
    v_time = time.time() - s
    

    # s = time.time()
    # frame_batch = []
    # for i in range(16):
    #     i_frame = np.array(Image.open('samples/%05d.jpg'%(i+1)))
    #     frame_batch.append(i_frame)
    # i_time = time.time() - s
    
    s = time.time()
    frame_batch = []
    for i in range(16):
        frame_batch.append(vr.get_batch([i]).asnumpy())
    i_time = time.time() - s

    print(v_time)
    print(i_time)

    # v_frame = vr[2].asnumpy()
    # v_frame = cv2.cvtColor(v_frame, cv2.COLOR_RGB2B   GR)
    # cv2.imshow('img_v', v_frame)
    # cv2.imshow('img_i', i_frame)
    # cv2.imshow('img_240', i_frame_240)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
