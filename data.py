import os
import torch
import numpy as np
from torchvision import datasets, transforms
from constants import GROUP_NUM, CLIENTNUM_GROUP, TOTAL_CLIENTS

def data_loaders(client_id: int, datasetPath: str, is_iid: bool, use_cuda: bool, batch_size: int, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    train_dataset = datasets.CIFAR10(os.path.join(datasetPath, 'cifar10'), train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
    test_dataset = datasets.CIFAR10(os.path.join(datasetPath, 'cifar10'), train=False, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),

    mytrain_virtuals = []
    mytest_virtuals = []

    label_num = 10
    train_sorted_idx_list = [[] for _ in range(label_num)]
    test_sorted_idx_list = [[] for _ in range(label_num)]

    for i in range(len(train_dataset)):
        class_idx = train_dataset[i][1]
        train_sorted_idx_list[class_idx].append(i)

    if not is_iid:
        label_num = 10
        train_sorted_idx_list = [[] for _ in range(label_num)]
        test_sorted_idx_list = [[] for _ in range(label_num)]
        for i in range(len(train_dataset)):
            class_idx = train_dataset[i][1]
            train_sorted_idx_list[class_idx].append(i)
        for i in range(len(test_dataset[0])):
            class_idx = test_dataset[0][i][1]
            test_sorted_idx_list[class_idx].append(i)
        mytrainstart_idx = client_id * (len(train_sorted_idx_list[0]) // GROUP_NUM)
        mytrainend_idx = (client_id + 1) * (len(train_sorted_idx_list[0]) // GROUP_NUM)
        myteststart_idx = client_id * (len(test_sorted_idx_list[0]) // GROUP_NUM)
        mytestend_idx = (client_id + 1) * (len(test_sorted_idx_list[0]) // GROUP_NUM)
        trainidx_step = (mytrainend_idx - mytrainstart_idx) // 3
        testidx_step = (mytestend_idx - myteststart_idx) // 3
        for virt_id in range(CLIENTNUM_GROUP):
            virtmytrainlabel1_idx = train_sorted_idx_list[virt_id][mytrainstart_idx : mytrainstart_idx + trainidx_step] #166data
            virtmytrainlabel2_idx = train_sorted_idx_list[(virt_id+1) % CLIENTNUM_GROUP][mytrainstart_idx + trainidx_step : mytrainstart_idx + trainidx_step*2 + 1] #167data
            virtmytrainlabel3_idx = train_sorted_idx_list[(virt_id+2) % CLIENTNUM_GROUP][mytrainstart_idx + trainidx_step*2 + 1 : mytrainend_idx] #167data

            virtmytestlabel1_idx = test_sorted_idx_list[virt_id][myteststart_idx : myteststart_idx + testidx_step] #33data
            virtmytestlabel2_idx = test_sorted_idx_list[(virt_id+1) % CLIENTNUM_GROUP][myteststart_idx + testidx_step : myteststart_idx + testidx_step*2] #33data
            virtmytestlabel3_idx = test_sorted_idx_list[(virt_id+2) % CLIENTNUM_GROUP][myteststart_idx + testidx_step*2 : mytestend_idx] #34data

            mytrain_virtuals.append(torch.utils.data.Subset(train_dataset, virtmytrainlabel1_idx + virtmytrainlabel2_idx + virtmytrainlabel3_idx))
            mytest_virtuals.append(torch.utils.data.Subset(test_dataset[0], virtmytestlabel1_idx + virtmytestlabel2_idx + virtmytestlabel3_idx))
    else:
        label_num = 10
        
        train_indices_virts = [np.array([],dtype=np.long) for _ in range(CLIENTNUM_GROUP)] #仮想workerそれぞれが担当するdataのidxを二重リストで。

        for indices_label in train_sorted_idx_list:#train setを抽出、分配
            splitted_indices = np.array_split(indices_label, TOTAL_CLIENTS)
            startidx = client_id*CLIENTNUM_GROUP
            for i in range(CLIENTNUM_GROUP): #各labelのdataを100分割して、実clientの部分から10人分抽出してそれぞれ追加
                train_indices_virts[i] = np.concatenate([splitted_indices[startidx+i], train_indices_virts[i]], 0)

        for virtid in range(CLIENTNUM_GROUP):
            mytrain_virtuals.append(torch.utils.data.Subset(train_dataset, train_indices_virts[virtid]))
        
    mytestdata = torch.utils.data.Subset(test_dataset[0], np.arange(client_id*(len(test_dataset[0])//GROUP_NUM), (client_id+1)*(len(test_dataset[0])//GROUP_NUM)))

    mytrain_loaders = [torch.utils.data.DataLoader(
        mytrain_virtuals[i], batch_size=batch_size, shuffle=True, **kwargs) for i in range(CLIENTNUM_GROUP)]
    mytest_loader = torch.utils.data.DataLoader(
        mytestdata, batch_size=test_batch_size, shuffle=True, **kwargs)
    
    return mytrain_loaders, mytest_loader