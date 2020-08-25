# action_recognition_online

Extracting videos into frames for training costs a huge amount of memory when the dataset has an overwholming amount of data. What is more, it is unfriendly for training new datasets for new tasks (increase data step by step) and with different resolutions. 

This repository could help you train your dataset based on videos and with fast speed (not calculating the accurate efficiency, but faster than cv2.VideoCapture)

# Results
The accuracies of three datasets HMDB, UCF-101, Kinetics are attached, which are based on 16f frame and 3D Resnext backbone. Results on HMDB and UCF-101 are obtained by fintuning the last block of 3D Resnext based on the pretrained model from https://github.com/craston/MARS. 


| Inputs  | HMDB (%) | UCF101 (%) | Kinetics 400 (%) | 
| --------| ----- | ----- | ----- |
| Frame  | 66.7 | 91.7 | 68.2 |
| Video  | 67.2 | 91.8 | 68.8 |

# How to use
1. Set your labels under the direction `data_process`
2. Create a reading function in `data_process/datasets`
3. Training your dataset with   
`python main.py --dataset your_dataset --modality RGB \
--n_classes 400 --n_finetune_classes 400 \
--batch_size 32 --log 1 --sample_duration 16 \
--model_name resnext --model_depth 101 --ft_begin_index 0 \
--pretrained_path "xxx.pth" \
--video_dir "your_data_path" \
--annotation_path "data_process/your_data_label" \
--result_path "results_kin400_pretrain_scratch/" \
--n_workers 4 --n_epochs '100'`

# Requirements
- python >= 3.6
- pytorch 1.3
- numpy
- [decord](https://github.com/dmlc/decord): `pip install decord`
- imutils:`pip install imutils`
- OpenCV 






