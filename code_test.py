import numpy as np

if __name__ == "__main__":
    total_frames = 900
    batch = []
    if total_frames >= 300:
        s_stamp = np.linspace(0, total_frames, int(300/16)+1)
        s_stamp = s_stamp.astype(np.int)
        for i in range(len(s_stamp[:-1])):
            i_batch = list(np.linspace(s_stamp[i], s_stamp[i+1]-1, 16).astype(np.int))
            batch.append(i_batch)
    import pdb; pdb.set_trace()

        
            
    #     import pdb; pdb.set_trace()
        
        
        
    #     clip_interval = int(total_frames / (300 / 16))
    #     s_stamp = range(0, total_frames, clip_interval)
    #     frame_interval = int(clip_interval / 15)

    #     for s in s_stamp[:-1]:
    #         i_batch = list(range(s, s+clip_interval, frame_interval))
    #         batch.append(i_batch)
    #     f_batch = list(range(total_frames-clip_interval, total_frames, frame_interval))
    #     batch.append(f_batch)
    # import pdb; pdb.set_trace()
        
        