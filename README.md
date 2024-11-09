# Environment

environment setting up command:
```sh
conda env create -f env.yaml
```


### Getting Datasets 

```sh
python ogbdataset.py
```

### Running

### Cora:
Modify the contents of the ortho_cn/Arg/args_tempo.py as

```sh
class Args:
    def __init__(self):
        self.use_valedges_as_input = True
        self.epochs = 150
        self.runs = 10
        self.dataset = "Cora"
        self.batch_size = 1152  
        self.testbs = 1152 
        self.maskinput = True
        self.mplayers = 1
        self.nnlayers = 3
        self.hiddim = 256
        self.ln = True
        self.lnnn = True
        self.res = False
        self.jk = True
        self.gnndp = 0.05
        self.xdp = 0.7
        self.tdp = 0.3
        self.gnnedp = 0.0
        self.predp = 0.05
        self.preedp = 0.4
        self.gnnlr = 0.0043
        self.prelr = 0.0024
        self.beta = 1
        self.alpha = 1.0
        self.use_xlin = True
        self.tailact = True
        self.twolayerlin = False
        self.increasealpha = False
        self.splitsize = -1
        self.probscale = 4.3
        self.proboffset = 2.8
        self.pt = 0.75
        self.learnpt = False
        self.trndeg = -1
        self.tstdeg = -1
        self.cndeg = -1
        self.predictor = 'cn1'
        self.depth = 1
        self.model = 'puregcn'
        self.save_gemb = False
        self.load = None
        self.loadmod = False
        self.savemod = False
        self.savex = False
        self.loadx = False
        self.cnprob = 0
        self.emb_dim=256
        self.K=8     #Hop you can modify

args = Args()
```

### Citeseer:
Modify the contents of the ortho_cn/Arg/args_tempo.py as

```sh
class Args:
    def __init__(self):
        self.use_valedges_as_input = True
        self.epochs = 150
        self.runs = 10
        self.dataset = "Citeseer"
        self.batch_size = 384  
        self.testbs = 384 
        self.maskinput = True
        self.mplayers = 1
        self.nnlayers = 3
        self.hiddim = 256
        self.ln = True
        self.lnnn = True
        self.res = False
        self.jk = True
        self.gnndp = 0.05
        self.xdp = 0.7
        self.tdp = 0.3
        self.gnnedp = 0.0
        self.predp = 0.05
        self.preedp = 0.4
        self.gnnlr = 0.0043
        self.prelr = 0.0024
        self.beta = 1
        self.alpha = 1.0
        self.use_xlin = True
        self.tailact = True
        self.twolayerlin = False
        self.increasealpha = False
        self.splitsize = -1
        self.probscale = 4.3
        self.proboffset = 2.8
        self.pt = 0.75
        self.learnpt = False
        self.trndeg = -1
        self.tstdeg = -1
        self.cndeg = -1
        self.predictor = 'cn1'
        self.depth = 1
        self.model = 'puregcn'
        self.save_gemb = False
        self.load = None
        self.loadmod = False
        self.savemod = False
        self.savex = False
        self.loadx = False
        self.cnprob = 0
        self.emb_dim=256
        self.K=8     #Hop you can modify

args = Args()
```

### Pubmed:
Modify the contents of the ortho_cn/Arg/args_tempo.py as

```sh
class Args:
    def __init__(self):
        self.use_valedges_as_input = True
        self.epochs = 150
        self.runs = 10
        self.dataset = "Pubmed"
        self.batch_size = 2048  
        self.testbs = 2048 
        self.maskinput = True
        self.mplayers = 1
        self.nnlayers = 3
        self.hiddim = 256
        self.ln = True
        self.lnnn = True
        self.res = False
        self.jk = True
        self.gnndp = 0.05
        self.xdp = 0.7
        self.tdp = 0.3
        self.gnnedp = 0.0
        self.predp = 0.05
        self.preedp = 0.4
        self.gnnlr = 0.0043
        self.prelr = 0.0024
        self.beta = 1
        self.alpha = 1.0
        self.use_xlin = True
        self.tailact = True
        self.twolayerlin = False
        self.increasealpha = False
        self.splitsize = -1
        self.probscale = 4.3
        self.proboffset = 2.8
        self.pt = 0.75
        self.learnpt = False
        self.trndeg = -1
        self.tstdeg = -1
        self.cndeg = -1
        self.predictor = 'cn1'
        self.depth = 1
        self.model = 'puregcn'
        self.save_gemb = False
        self.load = None
        self.loadmod = False
        self.savemod = False
        self.savex = False
        self.loadx = False
        self.cnprob = 0
        self.emb_dim=256
        self.K=8     #Hop you can modify

args = Args()
```


### collab:
Modify the contents of the ortho_cn/Arg/args_tempo.py as

```sh
class Args:
    def __init__(self):
        self.use_valedges_as_input = True
        self.epochs = 150
        self.runs = 10
        self.dataset = "collab"
        self.batch_size = 65536  
        self.testbs = 65536 
        self.maskinput = True
        self.mplayers = 1
        self.nnlayers = 3
        self.hiddim = 256
        self.ln = True
        self.lnnn = True
        self.res = False
        self.jk = True
        self.gnndp = 0.05
        self.xdp = 0.7
        self.tdp = 0.3
        self.gnnedp = 0.0
        self.predp = 0.05
        self.preedp = 0.4
        self.gnnlr = 0.0043
        self.prelr = 0.0024
        self.beta = 1
        self.alpha = 1.0
        self.use_xlin = True
        self.tailact = True
        self.twolayerlin = False
        self.increasealpha = False
        self.splitsize = -1
        self.probscale = 4.3
        self.proboffset = 2.8
        self.pt = 0.75
        self.learnpt = False
        self.trndeg = -1
        self.tstdeg = -1
        self.cndeg = -1
        self.predictor = 'cn1'
        self.depth = 1
        self.model = 'puregcn'
        self.save_gemb = False
        self.load = None
        self.loadmod = False
        self.savemod = False
        self.savex = False
        self.loadx = False
        self.cnprob = 0
        self.emb_dim=256
        self.K=8     #Hop you can modify

args = Args()
```

### ppa:
Modify the contents of the ortho_cn/Arg/args_tempo.py as

```sh
class Args:
    def __init__(self):
        self.use_valedges_as_input = True
        self.epochs = 150
        self.runs = 10
        self.dataset = "ppa"
        self.batch_size = 16384  
        self.testbs = 16384 
        self.maskinput = True
        self.mplayers = 1
        self.nnlayers = 3
        self.hiddim = 256
        self.ln = True
        self.lnnn = True
        self.res = False
        self.jk = True
        self.gnndp = 0.05
        self.xdp = 0.7
        self.tdp = 0.3
        self.gnnedp = 0.0
        self.predp = 0.05
        self.preedp = 0.4
        self.gnnlr = 0.0043
        self.prelr = 0.0024
        self.beta = 1
        self.alpha = 1.0
        self.use_xlin = True
        self.tailact = True
        self.twolayerlin = False
        self.increasealpha = False
        self.splitsize = -1
        self.probscale = 4.3
        self.proboffset = 2.8
        self.pt = 0.75
        self.learnpt = False
        self.trndeg = -1
        self.tstdeg = -1
        self.cndeg = -1
        self.predictor = 'cn1'
        self.depth = 1
        self.model = 'puregcn'
        self.save_gemb = False
        self.load = None
        self.loadmod = False
        self.savemod = False
        self.savex = False
        self.loadx = False
        self.cnprob = 0
        self.emb_dim=256
        self.K=8     #Hop you can modify

args = Args()
```

### citation2:
Modify the contents of the ortho_cn/Arg/args_tempo.py as

```sh
class Args:
    def __init__(self):
        self.use_valedges_as_input = True
        self.epochs = 150
        self.runs = 10
        self.dataset = "citation2"
        self.batch_size = 32768  
        self.testbs = 32768 
        self.maskinput = True
        self.mplayers = 1
        self.nnlayers = 3
        self.hiddim = 256
        self.ln = True
        self.lnnn = True
        self.res = False
        self.jk = True
        self.gnndp = 0.05
        self.xdp = 0.7
        self.tdp = 0.3
        self.gnnedp = 0.0
        self.predp = 0.05
        self.preedp = 0.4
        self.gnnlr = 0.0043
        self.prelr = 0.0024
        self.beta = 1
        self.alpha = 1.0
        self.use_xlin = True
        self.tailact = True
        self.twolayerlin = False
        self.increasealpha = False
        self.splitsize = -1
        self.probscale = 4.3
        self.proboffset = 2.8
        self.pt = 0.75
        self.learnpt = False
        self.trndeg = -1
        self.tstdeg = -1
        self.cndeg = -1
        self.predictor = 'cn1'
        self.depth = 1
        self.model = 'puregcn'
        self.save_gemb = False
        self.load = None
        self.loadmod = False
        self.savemod = False
        self.savex = False
        self.loadx = False
        self.cnprob = 0
        self.emb_dim=256
        self.K=8     #Hop you can modify

args = Args()
```

### ddi:
Modify the contents of the ortho_cn/Arg/args_tempo.py as

```sh
class Args:
    def __init__(self):
        self.use_valedges_as_input = True
        self.epochs = 150
        self.runs = 10
        self.dataset = "ddi"
        self.batch_size = 24576  
        self.testbs = 24576 
        self.maskinput = True
        self.mplayers = 1
        self.nnlayers = 3
        self.hiddim = 256
        self.ln = True
        self.lnnn = True
        self.res = False
        self.jk = True
        self.gnndp = 0.05
        self.xdp = 0.7
        self.tdp = 0.3
        self.gnnedp = 0.0
        self.predp = 0.05
        self.preedp = 0.4
        self.gnnlr = 0.0043
        self.prelr = 0.0024
        self.beta = 1
        self.alpha = 1.0
        self.use_xlin = True
        self.tailact = True
        self.twolayerlin = False
        self.increasealpha = False
        self.splitsize = -1
        self.probscale = 4.3
        self.proboffset = 2.8
        self.pt = 0.75
        self.learnpt = False
        self.trndeg = -1
        self.tstdeg = -1
        self.cndeg = -1
        self.predictor = 'cn1'
        self.depth = 1
        self.model = 'puregcn'
        self.save_gemb = False
        self.load = None
        self.loadmod = False
        self.savemod = False
        self.savex = False
        self.loadx = False
        self.cnprob = 0
        self.emb_dim=256
        self.K=8     #Hop you can modify

args = Args()
```

### if done,run:
```sh
python main.py
```
### Option2 :Also you can run Jupyternotebook after modifying class Args as above.
