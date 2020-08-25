import torch

if __name__ == "__main__":
    
    with open('data_process/Kinetics_labels/train.csv', 'r') as file:
        rows = file.readlines()[1:]
    new_file = []
    for line in rows:
        inform = line.split(',')
        new_line = '{}/{}_{:06d}_{:06d}.mp4'.format(inform[0], inform[1], 
                                        int(inform[2]), int(inform[3]))
        new_file.append(new_line)
    new_file = sorted(new_file)
    with open('data_process/Kinetics_labels/train_list.txt', 'w') as file:
        for line in new_file:
            file.write(line)
            file.write('\n')

    with open('data_process/Kinetics_labels/validate.csv', 'r') as file:
        rows = file.readlines()[1:]
    new_file = []
    for line in rows:
        inform = line.split(',')
        new_line = '{}/{}_{:06d}_{:06d}.mp4'.format(inform[0], inform[1], 
                                        int(inform[2]), int(inform[3]))
        new_file.append(new_line)
    new_file = sorted(new_file)
    with open('data_process/Kinetics_labels/val_list.txt', 'w') as file:
        for line in new_file:
            file.write(line)
            file.write('\n')

    with open('data_process/Kinetics_labels/test.csv', 'r') as file:
        rows = file.readlines()[1:]
    new_file = []
    for line in rows:
        inform = line.split(',')
        new_line = '{}/{}_{:06d}_{:06d}.mp4'.format(inform[0], inform[1], 
                                        int(inform[2]), int(inform[3]))
        new_file.append(new_line)
    new_file = sorted(new_file)
    with open('data_process/Kinetics_labels/test_list.txt', 'w') as file:
        for line in new_file:
            file.write(line)
            file.write('\n')

        