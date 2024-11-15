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




def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True):
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = []
    adjmask = torch.ones_like(source_edge, dtype=torch.bool)
    for perm in PermIterator(
            source_edge.device, source_edge.shape[0], batch_size
    ): 
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = torch.stack((source_edge[adjmask], target_edge[adjmask]), dim=0)
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   source_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        
        src, dst = source_edge[perm], target_edge[perm]

        edge = torch.stack((src, dst))

        adjs = CN_kij(adj,args.K,edge)

        pos_out = predictor.multidomainforward(h, adjs, torch.stack((src, dst)), cndropprobs=cnprobs, batch_idx=batch_idx + 1)

        pos_loss = -F.logsigmoid(pos_out).mean()

        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        neg_out = predictor(h, adj, torch.stack((src, dst_neg)))
        neg_loss = -F.logsigmoid(-neg_out).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss



@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()
    adj = data.full_adj_t
    h = model(data.x, adj)

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in PermIterator(source.device, source.shape[0], batch_size, False):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h,CN_kij(adj,args.K,torch.stack((src, dst))), torch.stack((src, dst))).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in PermIterator(source.device, source.shape[0], batch_size, False):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h,CN_kij(adj,args.K,torch.stack((src, dst))), torch.stack((src, dst))).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = 0.0 #test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr, h.cpu()





def main():
    args = Args()
    print(args, flush=True)
    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    writer.add_text("hyperparams", hpstr)

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    data, split_edge = loaddataset(args.dataset, False, args.load)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]
    
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor in ["incn1cn1", "sincn1cn1"]:
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []

    for run in range(args.runs):
        set_seed(run)
        bestscore = [0, 0, 0]
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp).to(device)

        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}])

        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput)
            print(f"trn time {time.time()-t1:.2f} s")
            if True:
                t1 = time.time()
                results = test(model, predictor, data, split_edge, evaluator,
                               args.testbs)
                results, h = results[:-1], results[-1]
                print(f"test time {time.time()-t1:.2f} s")
                writer.add_scalars(f"mrr_{run}", {
                        "trn": results[0],
                        "val": results[1],
                        "tst": results[2]
                    }, epoch)

                if True:
                    train_mrr, valid_mrr, test_mrr = results
                    train_mrr, valid_mrr, test_mrr = results
                    if valid_mrr > bestscore[1]:
                        bestscore = list(results) 
                        bestscore = list(results) 
                        if args.save_gemb:
                            torch.save(h, f"gemb/citation2_{args.model}_{args.predictor}.pt")

                    print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_mrr:.2f}%, '
                              f'Valid: {100 * valid_mrr:.2f}%, '
                              f'Test: {100 * test_mrr:.2f}%')
                    print('---', flush=True)
        print(f"best {bestscore}")
        if args.dataset == "citation2":
            ret.append(bestscore)
        else:
            raise NotImplementedError
    ret = np.array(ret)
    print(ret)
    print(f"Final result: {np.average(ret[:, 1])} {np.std(ret[:, 1])} {np.average(ret[:, 2])} {np.std(ret[:, 2])}")

if __name__ == "__main__":
    main()