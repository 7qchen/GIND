import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.instancenorm import _InstanceNorm

from torch_scatter import scatter
from torch_geometric.utils import to_undirected
from torch_geometric.utils import degree
from torch_geometric.typing import OptTensor


def cal_norm(edge_index, num_nodes=None, self_loop=False, cut=False):
    # calculate normalization factors: (2*D)^{-1/2}
    if num_nodes is None:
        num_nodes = edge_index.max()+1
        
    D = degree(edge_index[0], num_nodes)
    if self_loop:
        D = D + 1
    
    if cut:  # for symmetric adj
        D = torch.sqrt(1/D)
        D[D == float("inf")] = 0.
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        mask = row<col
        edge_index = edge_index[:,mask]
    else:
        D = torch.sqrt(1/2/D)
        D[D == float("inf")] = 0.
    
    if D.dim() == 1:
        D = D.unsqueeze(-1)

    return D, edge_index


##############################################################################################################
#
# Layer normalization. Modified from the original PyG's implementation of LayerNorm.
#
##############################################################################################################

class LayerNorm(torch.nn.Module):
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.empty((in_channels,)))
            self.bias = None
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.weight.size(0) >= 256:
                self.weight.data.fill_(0.5)
            else:
                self.weight.data.fill_(1.)

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
        if batch is None:
            out = x / (x.std(unbiased=False) + self.eps)

        else:
            batch_size = int(batch.max()) + 1

            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.mul_(x.size(-1)).view(-1, 1)

            var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                          reduce='add').sum(dim=-1, keepdim=True)
            var = var / norm

            out = x / (var + self.eps).sqrt().index_select(0, batch)

        if self.weight is not None:
            out = out * self.weight

        return out

##############################################################################################################
#
# Instance normalization. Modified from the original PyG's implementation of InstanceNorm.
#
##############################################################################################################

class InstanceNorm(_InstanceNorm):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=False):
        super().__init__(in_channels, eps, momentum, affine,
                         track_running_stats)

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
        if batch is None:
            out = x/(x.std(unbiased=False, dim=0) + self.eps)
        
        else:
            batch_size = int(batch.max()) + 1

            var = unbiased_var = x

            if self.training or not self.track_running_stats:
                norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
                norm = norm.view(-1, 1)
                unbiased_norm = (norm - 1).clamp_(min=1)

                var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                            reduce='add')

                unbiased_var = var / unbiased_norm
                var = var / norm

                momentum = self.momentum
                if self.running_var is not None:
                    self.running_var = (
                        1 - momentum
                    ) * self.running_var + momentum * unbiased_var.mean(0)
            else:
                if self.running_var is not None:
                    var = self.running_var.view(1, -1).expand(batch_size, -1)

            out = x / (var + self.eps).sqrt().index_select(0, batch)

        if self.weight is not None:
            out = out * self.weight.view(1, -1)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_features})'