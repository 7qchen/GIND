from __future__ import division
from __future__ import print_function

import os
import sys
import uuid
from time import perf_counter
import hydra
from omegaconf import OmegaConf

import torch
import torch.optim as optim

from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score

sys.path.append("../")

from lib.utils import set_seed, style
from lib.normalization import cal_norm

from model.gind import GIND

os.system("")


def train(model, device, train_loader, val_loader, test_loader, 
        add_self_loops, epochs, patience, lr, wd):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    checkpt_file = './pretrained/'+uuid.uuid4().hex+'_ppi.pt'
    print(checkpt_file)
    
    best_f1 = 0
    best_epoch = 0
    
    loss_fn = torch.nn.BCEWithLogitsLoss()

    t = perf_counter()
    loss_all = 0.
    total_examples = 0.
    test_lst = []
    for epoch in range(epochs):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            norm_factor, edge_index = cal_norm(data.edge_index, data.num_nodes, self_loop=add_self_loops, cut=True)
            output = model(data.x, edge_index, norm_factor, data.batch)

            loss = loss_fn(output, data.y)
            
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * data.num_nodes
            total_examples += data.num_nodes
        loss_all = loss_all/total_examples

        val_f1 = test(model, val_loader, add_self_loops, device)
        test_f1 = test(model, test_loader, add_self_loops, device)
        test_lst.append(test_f1)

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), checkpt_file)
        else:
            bad_counter += 1
        
        if bad_counter == patience:
            break
        
        print(style.WHITE + 'Epoch: {}, Loss_train: {:.7f}, '.format(epoch, loss_all))
        
        print(style.YELLOW + 'Val f1 is: {:.7f},'.format(val_f1), 
            'Best val f1 is: {:.4f} at epoch {},'.format(best_f1, best_epoch),
            'Test f1 is: {:.4f}.'.format(test_lst[best_epoch]) + style.WHITE)

    train_time = perf_counter()-t
    torch.save(test_lst, './test_result.pt')
    model.load_state_dict(torch.load(checkpt_file))
    test_f1 = test(model, test_loader, add_self_loops, device)

    return model, test_f1, train_time

def test(model, loader, add_self_loops, device):
    model.eval()

    ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            ys.append(data.y)
            data.to(device)
            norm_factor, edge_index = cal_norm(data.edge_index, data.num_nodes, self_loop=add_self_loops, cut=True)
            output = model(data.x, edge_index, norm_factor, data.batch)
            preds.append((output > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    dataset_name = conf.dataset
    params = conf.params[dataset_name]
    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'
    if conf.root == '':
        conf.root = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../../..")), 'data', 'PPI')
    print(OmegaConf.to_yaml(params))

    set_seed(conf.seed, device)
    if not os.path.exists('./pretrained/'):
        os.mkdir('./pretrained/')

    train_dataset = PPI(conf.root, split='train')
    val_dataset = PPI(conf.root, split='val')
    test_dataset = PPI(conf.root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    in_channels = train_dataset.num_features
    out_channels = train_dataset.num_classes

    model = GIND(
            in_channels=in_channels,
            out_channels=out_channels,
            **params.architecture).to(device)

    model, test_f1, train_time = train(model, device, train_loader, val_loader, test_loader, params.add_self_loops, 
                                    params.epochs, params.patience, params.lr, params.wd)

    print(style.RESET + "Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(train_time))
    print("Final result: {:.4f}".format(test_f1))

if __name__ == "__main__":
    main()