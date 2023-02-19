import torch
import os
import argparse
import numpy as np
import random
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import flwr as fl
from models import *

virtual_client_num = 100
client_num = 10
virt_per_real = virtual_client_num // client_num

def train(model, device, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(1, epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def test(model, device, criterion, test_loader):
	global client_id
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target)
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)
	acc = correct / len(test_loader.dataset)
	return test_loss, acc

def main():
    global args
    global client_num
    global client_id
    global locked_rate
    global g_locked_rate

    # Training settings
    parser = argparse.ArgumentParser(description='SPinS-FL simulator')
    parser.add_argument('client_id', type=int, choices=list(range(100)),
                            help='ID of the client. Choose from [0,1...,99]')
    parser.add_argument('base_seed', type=int, choices=list(range(100)),
                            help='base seed. Choose from [0,1...,99]')
    parser.add_argument('locked_rate', type=float, default=0.5,
                            help='locked_rate')
    parser.add_argument('is_iid', type=int, choices=list(range(2)),
                            help='when it is iid, set 1 otherwise 0')
    parser.add_argument('--g_locked_rate', type=float, default=0.1,
                            help='the base locked rate in the global model')
    parser.add_argument('--hp_seed', type=int, default=0,
                            help='seed used to select hyper parameter')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                            help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                            help='number of epochs to train (default: 14)')
    parser.add_argument('--load_checkpoint', type=bool, default=False,
                            help='whether you load checkpoint')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                            help='how sparse is each layer')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    client_id = args.client_id

    global is_iid
    if args.is_iid == 0:
        is_iid = False
    else:
        is_iid = True

    locked_rate = args.locked_rate
    g_locked_rate = args.g_locked_rate

    torch.cuda.set_device(client_id % 4)

    #torch.manual_seed(args.seed)
    torch.manual_seed(args.base_seed)
    random.seed(args.base_seed)
    np.random.seed(args.base_seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    train_dataset = datasets.CIFAR10(os.path.join(args.data, 'cifar10'), train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
    test_dataset = datasets.CIFAR10(os.path.join(args.data, 'cifar10'), train=False, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),


    traindatanum_client = int(len(train_dataset) / virtual_client_num)
    testdatanum_client = int(len(test_dataset[0]) / virtual_client_num) #only test set tensors are wrapped in 1-element-list.

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
        mytrainstart_idx = client_id * (len(train_sorted_idx_list[0]) // client_num)
        mytrainend_idx = (client_id + 1) * (len(train_sorted_idx_list[0]) // client_num)
        myteststart_idx = client_id * (len(test_sorted_idx_list[0]) // client_num)
        mytestend_idx = (client_id + 1) * (len(test_sorted_idx_list[0]) // client_num)
        trainidx_step = (mytrainend_idx - mytrainstart_idx) // 3
        testidx_step = (mytestend_idx - myteststart_idx) // 3
        for virt_id in range(virt_per_real):
            virtmytrainlabel1_idx = train_sorted_idx_list[virt_id][mytrainstart_idx : mytrainstart_idx + trainidx_step] #166data
            virtmytrainlabel2_idx = train_sorted_idx_list[(virt_id+1) % virt_per_real][mytrainstart_idx + trainidx_step : mytrainstart_idx + trainidx_step*2 + 1] #167data
            virtmytrainlabel3_idx = train_sorted_idx_list[(virt_id+2) % virt_per_real][mytrainstart_idx + trainidx_step*2 + 1 : mytrainend_idx] #167data

            virtmytestlabel1_idx = test_sorted_idx_list[virt_id][myteststart_idx : myteststart_idx + testidx_step] #33data
            virtmytestlabel2_idx = test_sorted_idx_list[(virt_id+1) % virt_per_real][myteststart_idx + testidx_step : myteststart_idx + testidx_step*2] #33data
            virtmytestlabel3_idx = test_sorted_idx_list[(virt_id+2) % virt_per_real][myteststart_idx + testidx_step*2 : mytestend_idx] #34data

            mytrain_virtuals.append(torch.utils.data.Subset(train_dataset, virtmytrainlabel1_idx + virtmytrainlabel2_idx + virtmytrainlabel3_idx))
            mytest_virtuals.append(torch.utils.data.Subset(test_dataset[0], virtmytestlabel1_idx + virtmytestlabel2_idx + virtmytestlabel3_idx))
    else:
        label_num = 10
        
        train_indices_virts = [np.array([],dtype=np.long) for _ in range(virt_per_real)] #仮想workerそれぞれが担当するdataのidxを二重リストで。

        for indices_label in train_sorted_idx_list:#train setを抽出、分配
            splitted_indices = np.array_split(indices_label, virtual_client_num)
            startidx = client_id*virt_per_real
            for i in range(virt_per_real): #各labelのdataを100分割して、実clientの部分から10人分抽出してそれぞれ追加
                train_indices_virts[i] = np.concatenate([splitted_indices[startidx+i], train_indices_virts[i]], 0)

        for virtid in range(virt_per_real):
            mytrain_virtuals.append(torch.utils.data.Subset(train_dataset, train_indices_virts[virtid]))
        
    mytestdata = torch.utils.data.Subset(test_dataset[0], np.arange(client_id*(len(test_dataset[0])//client_num), (client_id+1)*(len(test_dataset[0])//client_num)))

    train_loaders = [torch.utils.data.DataLoader(
        mytrain_virtuals[i], batch_size=args.batch_size, shuffle=True, **kwargs) for i in range(virt_per_real)]
    test_loader = torch.utils.data.DataLoader(
        mytestdata, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = SpinsConv6()

    if args.load_checkpoint:
        cp_location = "hp50narrowcifar10_conv6_200round.pt"
        checkpoint = torch.load(cp_location)
        print(f"caution!!! You are loading checkpoint from {cp_location}")
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad])
    criterion = nn.CrossEntropyLoss()
    if args.load_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    global score_without_bn_idx
    idx = 0
    for name, val in model.state_dict().items():
        if 'scores' in name or ('bn' in name and 'running' in name):
            if 'scores' in name:
                score_without_bn_idx.append(idx)
            idx += 1


    

    class LotteryClient(fl.client.NumPyClient):
        def get_parameters(self):
            score_bn_list = []
            for name, val in model.state_dict().items():
                if 'scores' in name or ('bn' in name and 'running' in name):
                    score_bn_list.append(val.cpu().numpy())
            return score_bn_list
        
        def global_lock(self, score_bn_list):
            global locked_rate
            global g_locked_rate
            global score_without_bn_idx
            global top_untracked_indices_list
            global bottom_untracked_indices_list
            global round_ctr

            layer_idx = 0
            for scoreidx in score_without_bn_idx: #killing global score
                v = score_bn_list[scoreidx]
                v_flatten = v.flatten()

                dead_indices = np.concatenate([top_untracked_indices_list[layer_idx], bottom_untracked_indices_list[layer_idx]], 0)

                all_indices = np.arange(len(v_flatten)) #score全体のindex配列
                live_mask = np.ones(all_indices.size, dtype=bool)
                live_mask[dead_indices] = False
                live_indices = all_indices[live_mask] #追跡中のindexだけからなる配列
                v_alive_flatten = v_flatten[live_mask] #現在追跡中のスコアのみからなる配列
                abs_lives_ranking = np.argsort(np.abs(v_alive_flatten))
                sorted_live_indices = live_indices[abs_lives_ranking] #追跡中のスコアのindexを、ランキング順に並べ替えたもの

                if live_indices.size // 2 - int((g_locked_rate / 2) * v.size) > v.size // 2 - int((locked_rate / 2) * v.size): #通常通りkillした時に、local unlock sectionまで侵さないようにする
                    bottomkilling_end = int((g_locked_rate / 2) * v.size)
                    topkilling_start = sorted_live_indices.size - bottomkilling_end #必ず上下同数ずつkillされる
                else:
                    bottomkilling_end = sorted_live_indices.size // 2 - int(((1.0 - locked_rate) / 2) * v.size) #local unlock sectionぎりぎりまで
                    topkilling_start = sorted_live_indices.size - bottomkilling_end #必ず上下同数ずつkillされる

                bottom_killed_indices = sorted_live_indices[:bottomkilling_end] #下位
                top_killed_indices = sorted_live_indices[topkilling_start:] #追跡をやめるスコア上位

                new_bottom_untracked = np.concatenate([bottom_untracked_indices_list[layer_idx], bottom_killed_indices], 0)
                new_top_untracked = np.concatenate([top_untracked_indices_list[layer_idx], top_killed_indices], 0)
                bottom_untracked_indices_list[layer_idx] = new_bottom_untracked
                top_untracked_indices_list[layer_idx] = new_top_untracked
                #if client_id == 0:
                    #print(f"round:{round_ctr}, layer:{layer_idx}")
                    #print(f"new bottom (current total:{new_bottom_untracked.size}):")
                    #print(new_bottom_untracked)
                    #print(f"new top (current total:{new_top_untracked.size}):")
                    #print(new_top_untracked)

                layer_idx += 1


        def set_parameters(self, score_bn_list):
            global locked_rate
            global g_locked_rate
            global score_without_bn_idx
            global top_untracked_indices_list
            global bottom_untracked_indices_list
            global round_ctr
            top_indices_list = []
            unlocked_indices_list = []
            bottom_indices_list = []

            layer_idx = 0
            for scoreidx in score_without_bn_idx: #locking local score
                v = score_bn_list[scoreidx]

                ###global lock support
                living_mask = np.ones(v.size, dtype=bool)
                all_indices = np.arange(v.size) #score全体のindex配列

                living_mask[top_untracked_indices_list[layer_idx]] = False
                living_mask[bottom_untracked_indices_list[layer_idx]] = False

                living_v_flatten = v.flatten()[living_mask]
                live_indices = all_indices[living_mask] #追跡中のindexだけからなる配列
                
                abs_livingscores_ranking = np.argsort(np.abs(living_v_flatten))
                sorted_live_indices = live_indices[abs_livingscores_ranking] #追跡中のスコアの大元のindexを、ランキング順に並べ替えたもの

                bottom_locked_end = living_v_flatten.size // 2 - int((1.0 - locked_rate) * v.size) // 2 #追跡中の個数から、unlockすべき個数を引けば、ロックする個数
                top_locked_start = living_v_flatten.size - bottom_locked_end

                bottom_locked_indices = np.concatenate([bottom_untracked_indices_list[layer_idx], sorted_live_indices[:bottom_locked_end]], 0)
                unlocked_indices = sorted_live_indices[bottom_locked_end:top_locked_start]
                top_locked_indices = np.concatenate([top_untracked_indices_list[layer_idx], sorted_live_indices[top_locked_start:]], 0)

                if client_id == 0:
                    if bottom_locked_indices.size != top_locked_indices.size or top_locked_indices.size != int(locked_rate / 2 * v.size) or v.size != bottom_locked_indices.size + unlocked_indices.size + top_locked_indices.size:
                        print("locked rate corruption!!")
                        print(f"round:{round_ctr}, layer:{layer_idx}")
                        print(f"total:{v.size}")
                        print(f"bottom:{bottom_locked_indices.size}")
                        print(f"unlock:{unlocked_indices.size}")
                        print(f"top:{top_locked_indices.size}")
                        print("")

                ###ここまで

                #abs_scores_ranking = np.argsort(np.abs(v.flatten()))
                #bottom_locked_end = int((locked_rate / 2) * v.size)
                #top_locked_start = int((1 - locked_rate / 2) * v.size)
                #bottom_locked_indices = abs_scores_ranking[:bottom_locked_end]
                #unlocked_indices = abs_scores_ranking[bottom_locked_end:top_locked_start]
                #top_locked_indices = abs_scores_ranking[top_locked_start:]
                top_indices_list.append(top_locked_indices)
                unlocked_indices_list.append(unlocked_indices)
                bottom_indices_list.append(bottom_locked_indices)
                layer_idx += 1

            parameters = []
            i = 0
            top_itr = 0
            unlocked_itr = 0
            bottom_itr = 0

            for name, val in model.state_dict().items():
                if 'scores' in name or ('bn' in name and 'running' in name):
                    parameters.append(score_bn_list[i])
                    i += 1
                elif 'top_locked' in name:
                    parameters.append(top_indices_list[top_itr])
                    top_itr += 1
                elif 'unlocked' in name:
                    parameters.append(unlocked_indices_list[unlocked_itr])
                    unlocked_itr += 1
                elif 'bottom_locked' in name:
                    parameters.append(bottom_indices_list[bottom_itr])
                    bottom_itr += 1
                else:
                    parameters.append(val)

            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            global round_ctr

            #if round_ctr % 10 == 9:
            if round_ctr % 10 == 0 and round_ctr != 0:
                if client_id == 0:
                    print(f"round_ctr:{round_ctr}")
                self.global_lock(parameters)

            self.set_parameters(parameters)

            train_loader = None

            if not is_iid:
                perm = torch.randperm(virt_per_real)
                selected_virtid = perm[client_id]
                train_loader = train_loaders[selected_virtid]
                #train_loader = train_loaders[(client_id+round_ctr) % virt_per_real]
                #round_ctr += 1
            else:
                virtids_participate = np.random.randint(0, virt_per_real-1, client_num)
                train_loader = train_loaders[virtids_participate[client_id]]
                #train_loader = train_loaders[random.randint(0, virt_per_real-1)]

            train(model, device,  train_loader, optimizer, criterion, args.epochs)
            round_ctr += 1
            return self.get_parameters(), len(train_loader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(model, device, criterion, test_loader)
            return float(loss), len(test_loader.dataset), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=LotteryClient())
    if args.save_model and client_id == 0:
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, "lock0.75myadamepseed0_400round.pt")




if __name__ == '__main__':
    main()
