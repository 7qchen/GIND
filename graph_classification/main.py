from __future__ import division
from __future__ import print_function

import os
import sys
import uuid
from time import perf_counter
import hydra
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.append("../")
from libs.utils import set_seed, style
from libs.normalization import cal_norm
from libs.optimization import LinearScheduler

from model.gind import GIND

os.system("")


def train(model, device, train_loader, test_loader, add_self_loops, epochs, lr, wd):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    scheduler = LinearScheduler(optimizer, epochs)

    checkpt_file = './pretrained/'+uuid.uuid4().hex+'.pt'
    print(checkpt_file)
    
    result = []
    total_iters = len(train_loader)
    t = perf_counter()

    for epoch in range(epochs):
        model.train()

        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            norm_factor, edge_index = cal_norm(data.edge_index, data.num_nodes, add_self_loops)
            output = model(data.x, edge_index, norm_factor, batch=data.batch)
            loss = F.cross_entropy(output, data.y)
            
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        test_acc = test(model, device, test_loader, add_self_loops)
        result.append(test_acc)

        train_loss = loss_all / total_iters
        print(style.WHITE + 'Epoch: {}, LR: {:.10f}, Train Loss: {:.4f}, '.format(epoch, scheduler.get_last_lr()[0], train_loss))
        train_acc = test(model, device, train_loader, add_self_loops)
        print(style.YELLOW + 'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(train_acc, test_acc))
        
        scheduler.step()
    
    train_time = perf_counter()-t
    
    return model, result, train_time

def test(model, device, loader, add_self_loops):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            norm_factor, edge_index = cal_norm(data.edge_index, data.num_nodes, add_self_loops)
            output = model(data.x, edge_index, norm_factor, batch=data.batch)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    dataset_name = conf.dataset
    params = conf.params[dataset_name]
    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'
    if conf.root == '':
        conf.root = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../../..")), 'data')
    print(OmegaConf.to_yaml(params))
    
    set_seed(conf.seed, device)

    dataset = TUDataset(conf.root, name=dataset_name).shuffle()
    num_fold = params.num_fold
    skf = StratifiedKFold(n_splits=num_fold, shuffle = True, random_state = conf.seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(dataset.data.y)), dataset.data.y):
        idx_list.append(idx)
    if dataset.num_features == 0:
        dataset.num_features = 1
    in_channels = dataset.num_features
    out_channels = dataset.num_classes
    
    results = [0.]*num_fold
    total_time = 0.

    if not os.path.exists('./pretrained/'):
        os.mkdir('./pretrained/')
    for fold_idx in range(num_fold):
        model = GIND(
            in_channels=in_channels,
            out_channels=out_channels,
            **params.architecture).to(device)

        idx_train, idx_test = idx_list[fold_idx]
        test_dataset = dataset[idx_test.tolist()]
        train_dataset = dataset[idx_train.tolist()]

        test_loader = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=0)
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=0)

        model, result, train_time = train(model, device, train_loader, test_loader,
                                        params.add_self_loops, params.epochs, params.lr, params.wd)
        
        results[fold_idx] = result  # [n_fold, epoch]
        total_time += train_time
    
    re_np = np.array(results)
    best_epoch = np.argmax(re_np.mean(0))
    re_all = [re_np[:,best_epoch].mean(), re_np[:,best_epoch].std()]
    
    print(style.BLUE + 'Graph classification mean accuracy and std are {}'.format(re_all))
    print(style.RESET + "Total time elapsed: {:.4f}s".format(total_time))

if __name__ == "__main__":
    main()