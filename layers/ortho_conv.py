from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
import torch as th


class NormalBasisConv(MessagePassing):
    def __init__(self, fixed=False, kwargs={}):
        super(NormalBasisConv, self).__init__()
        self.fixed = fixed
        if self.fixed:
            n_hidden = kwargs['n_hidden']
            self.register_buffer('three_term_relations', th.zeros(n_hidden, 3), persistent=False)
            self.fixed_relation_stored = False

    def message(self, x_j, norm):
        
        if isinstance(norm, SparseTensor):
            norm = norm.to_dense()
        return norm.unsqueeze(-1) * x_j

    def forward(self, edge_index,adj, last_h, second_last_h, inner_product_last=None, inner_product_second_last=None, is_training=True):
        '''if isinstance(adj, SparseTensor):
            row, col, norm = adj.coo()
            edge_index = th.stack([row, col], dim=0)
            if norm is None:
                norm = th.ones(row.size(0), device=row.device)
            else:
                norm = norm.to_dense()
        else:
            edge_index = (adj > 0).nonzero(as_tuple=False).t()
            norm = adj[edge_index[0], edge_index[1]]

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index should have shape (2, num_edges)")'''

        

        #rst = self.propagate(edge_index=edge_index, x=last_h, norm=adj.T)
        rst=adj  

        print(f"Shape of rst: {rst.shape}")


        '''orthogonalize'''
        if is_training:
            rst, inner_product_last, inner_product_second_last = self.orthogonalize(is_training,rst, last_h, second_last_h)
        else:
            rst, _, _ = self.orthogonalize(is_training,rst, last_h, second_last_h, inner_product_last, inner_product_second_last)

        th.cuda.empty_cache()
        return rst, inner_product_last, inner_product_second_last

    def orthogonalize(self,is_training, rst, last_h, second_last_h, inner_product_last=None, inner_product_second_last=None):
            """Orthogonalize `rst` wrt. `last_h` and `second_last_h`."""
            

            if isinstance(rst, SparseTensor):
                rst = rst.to_dense()
            if isinstance(last_h, SparseTensor):
                last_h = last_h.to_dense()
            if isinstance(second_last_h, SparseTensor):
                second_last_h = second_last_h.to_dense()
            

            if is_training:
                
                   
                inner_product_last = th.matmul(rst, last_h.T)
                

                rst = rst - last_h* inner_product_last
               
            else:
                inner_product_last = th.zeros(1, device=rst.device)


            if is_training:
                
        
                inner_product_second_last = th.matmul(rst, second_last_h.T)
            
               
               
                rst = rst - second_last_h * inner_product_second_last
            else:
                inner_product_second_last = th.zeros(1, device=rst.device)


            rst = rst / th.clamp((th.norm(rst,dim=0)),1e-8)
            
            return rst, inner_product_last, inner_product_second_last
