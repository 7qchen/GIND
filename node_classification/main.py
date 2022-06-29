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

sys.path.append("../")
from libs.data import get_data
from libs.utils import set_seed, style
from libs.normalization import cal_norm
from libs.metric import accuracy

from model.gind import GIND

os.system("")


def train(model, 
        edge_index, norm_factor, features, labels, 
        idx_train, idx_val, idx_test,
        epochs, patience, imp_lr, exp_lr, imp_wd, exp_wd):

    optimizer = optim.Adam([
                        {'params':model.params_imp,'weight_decay':imp_wd, 'lr': imp_lr},
                        {'params':model.params_exp,'weight_decay':exp_wd, 'lr': exp_lr}])
    
    checkpt_file = './pretrained/'+uuid.uuid4().hex+'.pt'
    print(checkpt_file)
    
    best_acc = 0
    best_loss = 1e5
    best_epoch = 0
    t = perf_counter()
    bad_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(features, edge_index, norm_factor)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        
        loss_train.backward()
        optimizer.step()

        acc_val, loss_val = test(model, features, edge_index, norm_factor, labels, idx_val)
        if acc_val >= best_acc:
            if loss_val <= best_loss:
                best_acc = acc_val
                best_epoch = epoch
                torch.save(model.state_dict(), checkpt_file)
            elif acc_val > best_acc:
                torch.save(model.state_dict(), checkpt_file)
                best_epoch = epoch
            
            best_acc = np.max((best_acc, acc_val))
            best_loss = np.min((best_loss, loss_val))
            bad_counter = 0
        else:
            bad_counter += 1
        
        if bad_counter == patience:
            break

        print(style.WHITE + 'Epoch: {:04d}'.format(epoch),
        'loss_train: {:.4f}'.format(loss_train.item()),
        "acc_train = {:.4f}".format(acc_train))

        print(style.YELLOW + 'acc_val is: {:.4f}, '.format(acc_val),
            'Best acc_val is: {:.4f} at epoch {}'.format(best_acc, best_epoch))
    
    train_time = perf_counter()-t
    
    model.load_state_dict(torch.load(checkpt_file))
    acc = test(model, features, edge_index, norm_factor, labels, idx_test)[0]
    return model, acc, train_time

def test(model, features, edge_index, norm_factor, labels, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, edge_index, norm_factor)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    if not isinstance(acc_test, float):
        acc_test = acc_test.item()
    loss_val = F.cross_entropy(output[idx_test], labels[idx_test])

    return acc_test, loss_val.item()

@hydra.main(config_path='conf', config_name='config')
def main(conf):
    dataset_name = conf.dataset
    params = conf.params[dataset_name]
    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'
    if conf.root == '':
        conf.root = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../../..")), 'data')
    print(OmegaConf.to_yaml(params))

    set_seed(conf.seed, device)

    edge_index, features, labels, in_channels, out_channels, \
    train_mask, val_mask, test_mask = get_data(conf.root, dataset_name, device)
    norm_factor, edge_index = cal_norm(edge_index, self_loop=params.add_self_loops)
    num_fold = params.num_fold
    
    results = [0.]*num_fold
    total_time = 0.

    if not os.path.exists('./pretrained/'):
        os.mkdir('./pretrained/')
    for fold_idx in range(num_fold):
        model = GIND(
            in_channels=in_channels,
            out_channels=out_channels,
            **params.architecture).to(device)

        idx_train, idx_val, idx_test = train_mask[:, fold_idx], val_mask[:, fold_idx], test_mask[:, fold_idx]

        model, result, train_time = train(model, edge_index, norm_factor, features, labels, 
                                        idx_train, idx_val, idx_test, params.epochs, params.patience, 
                                        params.imp_lr, params.exp_lr, params.imp_wd, params.exp_wd)
        
        results[fold_idx] = result
        total_time += train_time
    
    re_np = np.array(results)
    re_all = [re_np.mean(), re_np.std()]
    
    print(style.BLUE + 'Node classification mean accuracy and std are {}'.format(re_all))
    print(style.RESET + "Total time elapsed: {:.4f}s".format(total_time))

if __name__ == "__main__":
    main()