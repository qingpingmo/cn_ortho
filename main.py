import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter

import time
from typing import Iterable

from torch_geometric.utils import k_hop_subgraph
from datasets.loader import randomsplit,loaddataset
from layers.ortho_conv import NormalBasisConv
from layers.pure_conv import *
from models.GCN import GCN
from models.orthoNN import *
from utils.khop import PermIterator
from utils.sparse_culc import DropAdj
from Arg.args_tempo import Args,args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F

def CN_kij(adj: SparseTensor, k: int, ij: torch.Tensor, slice_size: int=512):
    device = adj.device()  
    ij = ij.to(device)  

    h = ij.shape[1] 
    
    n = adj.size(0)
    
    rets = []
    
   
    for idx in range(0, h, slice_size):
        tij = ij[:, idx: idx + slice_size]
        Ej = F.one_hot(tij[1].long(), num_classes=n).T.float().to(device) 
       
        Ei = F.one_hot(tij[0].long(), num_classes=n).float().to(device)   
     
        ret = []
       
        for _ in range(k): 
            Ej = adj @ Ej

            ret.append((Ei * Ej.T).sum(dim=-1))
           
        
        ret = torch.stack(ret, dim=-1)  # (slice_size, k)
       
        rets.append(ret)
      

    #(h, k)
    rets = torch.cat(rets, dim=0)



    adjs = [rets[:, i].unsqueeze(0) for i in range(k)]
   
    
    return adjs




def train(model, predictor, data, split_edge, optimizer, batch_size, maskinput=True, cnprobs=[], alpha=None):
    if alpha is not None:
        predictor.setalpha(alpha)

    model.train()
    predictor.train()
    pos_train_edge = split_edge['train']['edge'].to(data.x.device).t()


    total_loss = []
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])


    

    for batch_idx, perm in enumerate(PermIterator(pos_train_edge.device, pos_train_edge.shape[1], batch_size)):
        optimizer.zero_grad()

        adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei, sparse_sizes=(data.num_nodes, data.num_nodes)).to(pos_train_edge.device)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
            
        else:
            adj = data.adj_t


        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]

        adjs = CN_kij(adj,args.K,edge)

        
        pos_outs = predictor.multidomainforward(h, adjs, edge, cndropprobs=cnprobs, batch_idx=batch_idx + 1)
        pos_loss = -F.logsigmoid(pos_outs).mean()
        
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adjs, edge, cndropprobs=cnprobs, batch_idx=batch_idx + 1)
        neg_loss = -F.logsigmoid(-neg_outs).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        optimizer.step()
        print(f"[train] Loss: {loss.item()}")
        total_loss.append(loss.item())

       
        del adjmask, tei, adj, pos_outs, neg_outs, h, adjs
        torch.cuda.empty_cache()

    return np.mean(total_loss)


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    
    pos_train_pred = torch.cat([
        predictor(h, CN_kij(adj,args.K,pos_train_edge[perm].t()), pos_train_edge[perm].t(), is_training=False).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)


    pos_valid_pred = torch.cat([
        predictor(h, CN_kij(adj,args.K,pos_valid_edge[perm].t()), pos_valid_edge[perm].t(), is_training=False).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred = torch.cat([
        predictor(h, CN_kij(adj,args.K,neg_valid_edge[perm].t()), neg_valid_edge[perm].t(), is_training=False).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, CN_kij(adj,args.K,pos_test_edge[perm].t()), pos_test_edge[perm].t(), is_training=False).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred = torch.cat([
        predictor(h,  CN_kij(adj,args.K,neg_test_edge[perm].t()), neg_test_edge[perm].t(),is_training=False).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']

        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    return results, h.cpu()










def main():
    args = Args()
    print(args, flush=True)

    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
        evaluator = Evaluator(name=f'ogbl-ppa')
    else:
        evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load)
    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    
    ret = []

    for run in range(0, args.runs):
        set_seed(run)
        if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load) # get a new split of dataset
            data = data.to(device)
        bestscore = None
        
        
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        if args.loadx:
            with torch.no_grad():
                model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
            model.xemb[0].weight.requires_grad_(False)
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        if args.loadmod:
            keys = model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
        

        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}])
        
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if True:
                t1 = time.time()
                results, h = test(model, predictor, data, split_edge, evaluator,
                               args.testbs, args.use_valedges_as_input)
                print(f"test time {time.time()-t1:.2f} s")
                if bestscore is None:
                    bestscore = {key: list(results[key]) for key in results}
                for key, result in results.items():
                    writer.add_scalars(f"{key}_{run}", {
                        "trn": result[0],
                        "val": result[1],
                        "tst": result[2]
                    }, epoch)

                if True:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        if valid_hits > bestscore[key][1]:
                            bestscore[key] = list(result)
                            if args.save_gemb:
                                torch.save(h, f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                            if args.savex:
                                torch.save(model.xemb[0].weight.detach(), f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            if args.savemod:
                                torch.save(model.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                                torch.save(predictor.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---', flush=True)
        print(f"best {bestscore}")
        if args.dataset == "collab":
            ret.append(bestscore["Hits@50"][-2:])
        elif args.dataset == "ppa":
            ret.append(bestscore["Hits@100"][-2:])
        elif args.dataset == "ddi":
            ret.append(bestscore["Hits@20"][-2:])
        elif args.dataset == "citation2":
            ret.append(bestscore[-2:])
        elif args.dataset in ["Pubmed", "Cora", "Citeseer"]:
            ret.append(bestscore["Hits@100"][-2:])
        else:
            raise NotImplementedError
    ret = np.array(ret)
    print(ret)
    print(f"Final result: val {np.average(ret[:, 0]):.4f} {np.std(ret[:, 0]):.4f} tst {np.average(ret[:, 1]):.4f} {np.std(ret[:, 1]):.4f}")


if __name__ == "__main__":
    main()