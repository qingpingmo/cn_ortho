import torch.nn as nn 
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch import Tensor
import torch_sparse
from typing import List, Tuple, Final


class PermIterator:
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(size, device=device) if training else torch.arange(size, device=device)
        #print(f"[PermIterator] Initialized with size={size}, bs={bs}, device={device}")

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) * (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret


def generate_base_adj(data, split_edge, maskinput, batch_size):
    pos_train_edge = split_edge['train']['edge'].to(data.x.device).t()
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)

    #print(f"[generate_base_adj] pos_train_edge shape: {pos_train_edge.shape}")
    #print(f"[generate_base_adj] data.x.device: {data.x.device}")

    for perm in PermIterator(adjmask.device, adjmask.shape[0], batch_size):
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(
                tei, sparse_sizes=(data.num_nodes, data.num_nodes)
            ).to(pos_train_edge.device) 
            #print(f"[generate_base_adj] Sparse adj shape: {adj.sizes()}")
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
            #print(f"[generate_base_adj] Using full adj_t")
    return adj


def generate_adjs_for_k(data, split_edge, K, args, maskinput, batch_size):
    
    adj_list = []  

    if maskinput:
        
        pos_train_edge = split_edge['train']['edge'].to(data.x.device).t()
        
        for k in range(1, K + 1):
            adj_k_hop_list = []
            for perm in PermIterator(pos_train_edge.device, pos_train_edge.shape[1], batch_size):
                adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
                adjmask[perm] = 0
                tei = pos_train_edge[:, adjmask]

                adj = SparseTensor.from_edge_index(
                    tei, sparse_sizes=(data.num_nodes, data.num_nodes)
                ).to(pos_train_edge.device)
                adj = adj.to_symmetric()
                
                #k-hop adj
                adj_k_hop = generate_adj_k_hop(adj, k)  
                adj_k_hop_list.append(adj_k_hop)
            
            adj_list.append(adj_k_hop_list)

    else:
        
        base_adj = data.adj_t  

        
        for k in range(1, K + 1):
            adj_k_hop = generate_adj_k_hop(base_adj, k)
            adj_list.append(adj_k_hop)

    return adj_list 




def generate_adj_0_1_hop(adj_1):
    
    device = adj_1.device()  

    

    try:
        loop_edge = torch.arange(adj_1.size(0), dtype=torch.int64, device=device)
        loop_edge = torch.stack([loop_edge, loop_edge], dim=0)  
        
    except TypeError as e:
        
        raise

    return SparseTensor.from_edge_index(loop_edge).to(device) 



def generate_adj_k_hop(adj, k):

    result_adj = adj
    for _ in range(k - 1):  
        result_adj = result_adj.matmul(adj)
        
    return result_adj