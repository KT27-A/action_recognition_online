import os
from decord import VideoReader


if __name__ == "__main__":
    data_type = 'val'
    file_path = 'data_process/Kinetics_labels/{}_list.txt'.format(data_type)
    video_dir = '/media/alien/Training/Kinetics/kinetics'
    annotation = 'kinetics_400_{}'.format(data_type)
    with open(file_path, 'r') as f:
        videos = f.readlines()
    for v in videos:
        video_path = os.path.join(video_dir, annotation, v.strip('\n'))
        if os.path.exists(video_path):
            try: vr = VideoReader(video_path)
            except:
                print('video path {} cannot be opened'.format(video_path))
                with open('not_open_file_{}.txt'.format(data_type), 'a') as f:
                    f.write(video_path)
                    f.write('\n')
        else:
            print('video path {} does not exists'.format(video_path))
            with open('not_exist_file_{}.txt'.format(data_type), 'a') as f:
                f.write(video_path)
                f.write('\n')
        
        