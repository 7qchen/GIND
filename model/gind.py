from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add

import sys
sys.path.append("lib/")

from libs.normalization import LayerNorm, InstanceNorm
from libs.optimization import VariationalHidDropout


def get_act(act_type):
    act_type = act_type.lower()
    if act_type == 'identity':
        return nn.Identity()
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'elu':
        return nn.ELU(inplace=True)
    elif act_type == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError

@torch.enable_grad()
def regularize(z, x, reg_type, edge_index=None, norm_factor=None):
    z_reg = norm_factor*z

    if reg_type == 'Lap':  # Laplacian Regularization
        row, col = edge_index
        loss = scatter_add(((z_reg.index_select(0, row)-z_reg.index_select(0, col))**2).sum(-1), col, dim=0, dim_size=z.size(0))
        return loss.mean()
    
    elif reg_type == 'Dec':  # Feature Decorrelation
        zzt = torch.mm(z_reg.t(), z_reg)
        Dig = 1./torch.sqrt(1e-8+torch.diag(zzt, 0))
        z_new = torch.mm(z_reg, torch.diag(Dig))
        zzt = torch.mm(z_new.t(), z_new)
        zzt = zzt - torch.diag(torch.diag(zzt, 0))
        zzt = F.hardshrink(zzt, lambd = 0.5)
        square_loss = F.mse_loss(zzt, torch.zeros_like(zzt))
        return square_loss

    else:
        raise NotImplementedError


class Append_func(nn.Module):
    def __init__(self, coeff, reg_type):
        super().__init__()
        self.coeff = coeff
        self.reg_type = reg_type

    def forward(self, z, x, edge_index, norm_factor):
        if self.reg_type == '' or self.coeff == 0.:
            return z
        else:
            z = z if z.requires_grad else z.clone().detach().requires_grad_()
            reg_loss = regularize(z, x, self.reg_type, edge_index, norm_factor)
            grad = autograd.grad(reg_loss, z, create_graph=True)[0]
            z = z - self.coeff * grad
            return z


class MLP(nn.Module):
    def __init__(self, c_in, c_out, middle_channels, hidden_act='relu', out_act='identity', dropout=0.):
        super(MLP, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.middle_channels = middle_channels

        self.hidden_act = get_act(hidden_act)
        self.out_act = get_act(out_act)

        c_ins = [c_in] + middle_channels
        c_outs = middle_channels + [c_out]
        
        self.lins = nn.ModuleList()
        for _, (in_dim, out_dim) in enumerate(zip(c_ins, c_outs)):
            self.lins.append(nn.Linear(int(in_dim), int(out_dim)))

        self.drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, xs):
        if len(self.lins) > 1:
            for _, lin in enumerate(self.lins[:-1]):
                xs = lin(xs)
                xs = self.hidden_act(xs)
                xs = self.drop(xs)
            xs = self.lins[-1](xs)
            xs = self.out_act(xs)
        
        else:
            xs = self.drop(xs)
            xs = self.lins[-1](xs)

        return xs


class Implicit_Func(nn.Module):
    def __init__(self, hidden_channel, middle_channel, alpha, norm, dropout, act, double_linear, rescale):
        super().__init__()
        self.alpha = alpha
        self.W = nn.Linear(hidden_channel, hidden_channel, bias=False)

        self.double_linear = double_linear
        if self.double_linear:
            self.U = nn.Linear(hidden_channel, middle_channel)

        self.norm = eval(norm)(middle_channel)
        
        self.rescale = rescale
        
        self.act = get_act(act)
        
        self.drop = VariationalHidDropout(dropout)
    
    def _reset(self, z):
        self.drop.reset_mask(z)
       
    def forward(self, z, x, edge_index, norm_factor, batch):
        num_nodes = x.size(0)
        row, col = edge_index
        
        if self.rescale:
            degree = 1./norm_factor
            degree[degree == float("inf")] = 0.
        else:
            degree = 1.

        if self.double_linear:
            WzUx = self.W(z) + degree * self.U(x)
        else:
            WzUx = self.W(z + degree * x)

        WzUx = norm_factor * WzUx
        WzUx = WzUx.index_select(0, row) - WzUx.index_select(0, col)

        if batch is not None:
            WzUx = self.norm(self.act(WzUx), batch.index_select(0, row))
        else:
            WzUx = self.norm(self.act(WzUx))
        
        new_z = scatter_add(WzUx*norm_factor[row], row, dim=0, dim_size=num_nodes)
        new_z -= scatter_add(WzUx*norm_factor[col], col, dim=0, dim_size=num_nodes)

        new_z = -F.linear(new_z, self.W.weight.t())
    
        z = self.alpha * self.drop(new_z) + (1 - self.alpha) * z
        
        return z


class Implicit_Module(nn.Module):
    def __init__(self, hidden_channel, middle_channels, alpha, norm, dropout, act, double_linear, rescale):
        super().__init__()
        Fs = [Implicit_Func(hidden_channel, middle_channel, alpha, norm, dropout, act, double_linear, rescale) for middle_channel in middle_channels]
        self.Fs = nn.ModuleList(Fs)
    
    def _reset(self, z):
        for func in self.Fs:
            func._reset(z)

    def forward(self, z, x, edge_index, norm_factor, batch):
        for func in self.Fs:
            z = func(z, x, edge_index, norm_factor, batch)
        return z


class GIND(nn.Module):
    r"""The Graph Neural Network from the "Optimization-Induced Graph Implicit 
    Nonlinear " paper.

    Args:
        hidden_channels (int): The number of hidden channels of the equilibrium.
        num_layers (int): The number of layers in the implicit module. Note that 
            our model only has one implicit module.
        alpha (float): The hyper-parameter introduced in Eq. (14) in the paper. 
        iter_nums: (Tuple[int, int]): The hyper-paprameters used for Phantom Gradient. 
            The first number is the total iteration number, while the second one 
            is the iteration number that requires gradient computation, i.e., the 
            unrolling step.
        norm (str): The sign-preserving normalization layer introduced in Eq. (18).
        rescale (bool): Setting rescale=True corresponds to the symmetric normalization 
            variant (Eq. (9)), while rescale=False corresponds to the row-normalization 
            variant (Eq. (34)). 
        linear (bool): Whether to use a linear layer or MLP for the output function. 
        double_linear (bool): Whether to use double matrix multiplications for the affine 
            transformation on the input feature matrix $X$ (see Eq. (9a)).
        reg_type (str) and reg_coeff (float): The hyper-parameters introduced in 
            optimization-inspired feature regularization in the paper.
        final_reduce (str): The read-out function used for graph-level tasks. 
    """
    def __init__(self, 
        in_channels: int, hidden_channels: int, out_channels: int, 
        num_layers: int, alpha: float, iter_nums: Tuple[int, int], 
        dropout_imp: float = 0., dropout_exp: float = 0.,
        drop_input: bool = False, norm: str = 'LayerNorm',
        residual: bool = True, rescale: bool = True,
        linear: bool = True, double_linear: bool = True, 
        act_imp: str = 'tanh', act_exp: str = 'elu',
        reg_type: str = '', reg_coeff: float = 0., 
        final_reduce: str = ''):
        super().__init__()

        self.total_num, self.grad_num = iter_nums
        self.no_grad_num = self.total_num - self.grad_num
        
        self.reg_type = reg_type
        self.dropout_exp = dropout_exp
        self.act = get_act(act_exp)
        self.residual = residual
        self.rescale = rescale

        self.drop_input = drop_input
        
        self.extractor = nn.Linear(in_channels, hidden_channels)
        
        middle_channels = [hidden_channels]*num_layers
        self.implicit_module = Implicit_Module(hidden_channels, middle_channels, 
                                            alpha, norm, dropout_imp, act_imp, 
                                            double_linear, rescale)
        self.Append = Append_func(coeff=reg_coeff, reg_type=reg_type)

        if linear:
            mlp_params = {'c_in': hidden_channels, 
                        'c_out': out_channels,
                        'middle_channels': [],
                        'hidden_act': act_exp,
                        'dropout': dropout_exp}
        else:
            mlp_params = {'c_in': hidden_channels, 
                        'c_out': out_channels,
                        'middle_channels': [hidden_channels],
                        'hidden_act': act_exp,
                        'dropout': dropout_exp}
        self.last_layer = MLP(**mlp_params)
        self.reduce = final_reduce

        self.init_weights()
        
        self.params_imp = list(self.implicit_module.parameters())
        self.params_exp = list(self.extractor.parameters()) + list(self.last_layer.parameters()) + list(self.Append.parameters())
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)

        nn.init.xavier_normal_(self.last_layer.lins[-1].weight, gain=1.)
        
    def multiple_steps(self, iter_start, iter_num, z, x, edge_index, norm_factor, batch):
        for _ in range(iter_start, iter_start+iter_num):
            z = self.Append(z, x, edge_index=edge_index, norm_factor=norm_factor)
            z = self.implicit_module(z, x, edge_index, norm_factor, batch)
        return z

    def forward(self, x, edge_index, norm_factor, batch=None):
        if self.drop_input:
            x = F.dropout(x, self.dropout_exp, training=self.training)

        x = self.extractor(x)

        self.implicit_module._reset(x)

        z = torch.zeros_like(x)
        with torch.no_grad():
            z = self.multiple_steps(0, self.no_grad_num, z, x, edge_index, norm_factor, batch)
        new_z = self.multiple_steps(self.no_grad_num-1, self.grad_num, z, x, edge_index, norm_factor, batch)

        if self.rescale:
            z = norm_factor*new_z + x if self.residual else new_z
        else:
            z = new_z + x if self.residual else new_z

        if self.reduce == 'add':
            z = global_add_pool(z, batch)

        pred = self.last_layer(z)
        return pred