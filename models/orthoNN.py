import itertools
import torch as th
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
from typing import Iterable
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from torch_sparse import SparseTensor
import torch_sparse
from torch_scatter import scatter_add
from typing import Iterable, Final


import sys
sys.path.append("..")
from utils.sparse_culc import DropAdj
from layers.ortho_conv import NormalBasisConv
from layers.pure_conv import *
from models.orthoNN import *
from utils.sparse_culc import DropAdj
from Arg.args_tempo import args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNLinkPredictor(nn.Module):
    cndeg: int
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta * th.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                  nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_channels, hidden_channels),
                                  lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(nn.Linear(1, hidden_channels),
                                    nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_channels, hidden_channels),
                                    lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
                                    nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        
        self.xijlin = nn.Sequential(nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
                                    nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        
       
        self.mlp_sparse = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),  
            lnfn(hidden_channels, ln),  
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity()
        )

  
        self.mlp_fusion = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),  
            lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
            lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
            nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
            nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
            nn.Linear(hidden_channels, out_channels) 
        )


        self.cndeg = cndeg
        self.K = args.K
        self.convs = nn.ModuleList([NormalBasisConv() for _ in range(self.K)])
        self.ln=ln

        self.accumulated_inner_products = []
        self.batch_count = 0
        self.hidden_channels = hidden_channels
        self.current_batch_progress = 0  
        self.batch_size = None
        self.dynamic_xcnlin = None
        self.dropout=dropout
        self.tailact=tailact

    def initialize_xcnlin(self, adj_size):
        """Initialize self.xcnlin based on adjs[0].size(1)."""
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if self.ln else nn.Identity()
        if self.dynamic_xcnlin is None: 
            in_channels = adj_size[1]
            lnfn = lambda dim: nn.LayerNorm(dim) if self.ln else nn.Identity()
            self.dynamic_xcnlin = nn.Sequential(
                nn.Linear(in_channels, self.hidden_channels),
                nn.Dropout(self.dropout, inplace=True),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_channels, self.hidden_channels),
                lnfn(self.hidden_channels),
                nn.Dropout(self.dropout, inplace=True),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_channels, self.hidden_channels) if not self.tailact else nn.Identity()
            ).to(device)

    def reset_constants(self, device, size_x):
        
        self.accumulated_inner_products = []
        #self.batch_count = 0
        #self.current_batch_progress = 0

    def accumulate_constants(self, inner_product_sums, batch_size, device, batch_id):
        

        
        while len(self.accumulated_inner_products) <= batch_id:
            self.accumulated_inner_products.append([th.zeros_like(inner_product_sums[0][j], device=device) for j in range(2 * self.K)])

        for j in range(2 * self.K):

            
            if batch_id > 0:
               
                self.accumulated_inner_products[batch_id][j] = self.accumulated_inner_products[batch_id-1][j].to(device) + \
                    (1 / batch_id) * (inner_product_sums[batch_id][j] - self.accumulated_inner_products[batch_id-1][j])
               
            else:
                
                self.accumulated_inner_products[batch_id][j] = inner_product_sums[batch_id][j]

 


    def multidomainforward(self, x, adjs, tar_ei, filled1: bool = False, cndropprobs: Iterable[float] = [], batch_idx=None, is_training=True):
        
        device = x.device
        if self.batch_size is None:
            self.batch_size = tar_ei.size(1)

        if self.dynamic_xcnlin is None:
            self.initialize_xcnlin(adjs[0].size())

        if any(isinstance(adj, list) for adj in adjs):
            adjs = list(itertools.chain.from_iterable(adjs))

        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij = self.xijlin(xi * xj)

        x = x + self.xlin(x)

        
        cn_k=[]
        for k in range(args.K):
            cn_k.append(adjs[k])

        sparse_diffs=[]
        for k in range(args.K):
            sparse_diffs.append(cn_k[k])

       
        h0 = sparse_diffs[0] / th.clamp((th.norm(sparse_diffs[0],dim=0)), 1e-8)
    
        inner_product_sums = []
        if batch_idx is not None:
            while len(inner_product_sums) <= batch_idx:
                inner_product_sums.append([th.zeros((1, 1), device=device) for _ in range(2 * self.K)])
       
        second_last_h = th.zeros_like(h0, device=device)
        last_h = h0

        for i, (con, adj) in enumerate(zip(self.convs, sparse_diffs[1:]), 1):
            adj = adj.to(device)
            h_i = None  

            if is_training:
                h_i, inner_product_last, inner_product_second_last = con(tar_ei,adj, last_h, second_last_h, is_training=True)
               
                
                if batch_idx is not None:
                    inner_product_sums[batch_idx][2 * (i - 1)] += inner_product_last.to(device)
                   
                    inner_product_sums[batch_idx][2 * (i - 1) + 1] += inner_product_second_last.to(device)
                    
            else:
                
                inner_product_last = self.accumulated_inner_products[2][2 * (i - 1)].to(device)
               
                inner_product_second_last = self.accumulated_inner_products[2][2 * (i - 1) + 1].to(device)
               
                h_i, _, _ = con(adj, last_h, second_last_h, inner_product_last, inner_product_second_last, is_training=False)

            if h_i is not None:
                second_last_h = last_h
                last_h = h_i.to(device)
              
                sparse_diffs[i] = h_i.to(device)


        if is_training:
            self.accumulate_constants(inner_product_sums, tar_ei.size(1), device, batch_idx)
          

        
        xs = []

        for xcn in sparse_diffs[k]:
           
            xcn_expanded = xcn.unsqueeze(-1).repeat(1, self.hidden_channels)  # (1, h) -> (h, hidden_channel)
            xcn_transformed = self.mlp_sparse(xcn_expanded.to(device))
            fused = xcn_transformed * self.beta + xij  
            fusion_transformed = self.mlp_fusion(fused.to(device))
            xs.append(fusion_transformed)

        xs = torch.cat(xs, dim=-1)

        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False, is_training=True):
        return self.multidomainforward(x, adj, tar_ei, filled1, [], None, is_training)


predictor_dict = {
    
    "cn1": CNLinkPredictor

}
