def remove_not_exist_open():
    data_type = 'val'
    with open('not_exist_file_{}.txt'.format(data_type), 'r') as f:
        not_exist_data = f.readlines()
    with open('not_open_file_{}.txt'.format(data_type), 'r') as f:
        not_open_data = f.readlines()
    with open('data_process/Kinetics_labels/{}_list.txt'.format(data_type), 'r') as f:
        ori_data = f.readlines()
    for element in not_exist_data:
        video = element.split('kinetics_400_{}/'.format(data_type))[-1]
        if video in ori_data:
            ori_data.remove(video)
    for element in not_open_data:
        video = element.split('kinetics_400_{}/'.format(data_type))[-1]
        if video in ori_data:
            ori_data.remove(video)
    with open('{}_remove.txt'.format(data_type), 'w') as f:
        f.writelines(ori_data)

def remove_not_open():
    with open('not_open_file.txt', 'r') as f:
        not_exists_data = f.readlines()
    with open('test_remove.txt', 'r') as f:
        ori_data = f.readlines()
    for element in not_exists_data:
        video = element.split('kinetics_400_test/')[-1]
        if video in ori_data:
            ori_data.remove(video)
    with open('test_remove.txt', 'w') as f:
        # for element in ori_data:
        #     f.write(element)
        f.writelines(ori_data)


if __name__ == "__main__":
    remove_not_exist_open()