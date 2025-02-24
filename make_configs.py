import yaml
from itertools import product
import copy
import os

# delete old files first
for f in os.listdir("configs/models"):
    if "default" not in f:
        os.remove("configs/models/"+f)
for f in os.listdir("configs/datasets"):
    if "default" not in f:
        os.remove("configs/datasets/"+f)
for f in os.listdir("configs/train"):
    if "default" not in f:
        os.remove("configs/train/"+f)
for f in os.listdir("configs/opt"):
    if "default" not in f:
        os.remove("configs/opt/"+f)

# Make model configs
Ls = [1,2,3,4,5,6,7,8]
Hs = [1,2,3,4,5,6,7,8]
Ds = [1,2,4,8,16,32,64,128,256]
dropouts = [0.0,0.05,0.1]
use_tokenize = [True,False]
use_rope = [True,False]

with open('configs/models/default_model.yaml',"r") as fin:
    default_model_config = yaml.load(fin,Loader=yaml.FullLoader)
for p in product(Ls,Hs,Ds,dropouts,use_tokenize,use_rope):
    L,H,D,drop,tok,rope = p

    model_name = "tok" if tok else "mse"
    model_name += f"_L{L}H{H}D{D}"
    if drop > 0:
        model_name += f"_drop{drop}"
    if rope:
        model_name += "_rope"
    
    config = copy.deepcopy(default_model_config)
    config['name'] = model_name
    config['GPT']['n_layer'] = L
    config['GPT']['n_head'] = H
    config['GPT']['n_embd'] = D
    config['GPT']['dropout'] = drop
    config['GPT']['tokenized'] = tok
    config['GPT']['use_rope'] = rope

    with open("configs/models/"+model_name+".yaml","w") as fout:
        yaml.dump(config,fout)

# Make dataset configs
## settings for frequency
wRanges = [(1,5), (1,3), (3,5), [(1,2.5), (3.5,5)]]
wRangeLabels = ['1to5','1to3','3to5','1to2.5U3.5to5']
wLinspaces = [(1,5,2**k) for k in range(4,13)] +\
             [(1,3,2**k) for k in range(4,13)] +\
             [(3,5,2**k) for k in range(4,13)] +\
             [[(1,2.5,(2**k)/2), (3.5,5,(2**k)/2)] for k in range(4,13)]
wLinspaceLabels = ['1to5N'+str(2**k) for k in range(4,13)] +\
                  ['1to3N'+str(2**k) for k in range(4,13)] +\
                  ['3to5N'+str(2**k) for k in range(4,13)] +\
                  ['1to2.5U3.5to5N'+str(2**k) for k in range(4,13)]
wDiscrete = [
    [1,2,3,4,5],
    [1,2,3],
    [3,4,5],
    [1,2,4,5]
]
wDiscreteLabels = ['12345','123','345','1245']
ws = wRanges + wLinspaces + wDiscrete
wLabels = wRangeLabels + wLinspaceLabels + wDiscreteLabels

## settings for damping
betaRanges = [(0,5), (0,2.5), (2.5,5), [(0,2), (3,5)]]
betaRangeLabels = ['0to5','0to2.5','2.5to5','0to2U3to5']
betaLinspaces = [(0,5,2**k) for k in range(4,13)] +\
                [(0,2.5,2**k) for k in range(4,13)] +\
                [(2.5,5,2**k) for k in range(4,13)] +\
                [[(0,2,(2**k)/2), (3,5,(2**k)/2)] for k in range(4,13)]
betaLinspaceLabels = ['0to5N'+str(2**k) for k in range(4,13)] +\
                    ['0to2.5N'+str(2**k) for k in range(4,13)] +\
                    ['2.5to5N'+str(2**k) for k in range(4,13)] +\
                    ['0to2U3to5N'+str(2**k) for k in range(4,13)]
betaDiscrete = [0,1,2,3,4,5]
betaDiscreteLabels = ['0','1','2','3','4','5']
betas = betaRanges + betaLinspaces + betaDiscrete
betaLabels = betaRangeLabels + betaLinspaceLabels + betaDiscreteLabels

## settings for pin_amplitude
pin_amp = [None,1.0]

## settings for instance_norm
inst_norm = [True,False]

## settings for dt
dts = [0.1,0.05]

with open("configs/datasets/default_dataset.yaml","r") as fin:
    default_dataset_config = yaml.load(fin,Loader=yaml.FullLoader)
for p in product(ws,betas,pin_amp,inst_norm,dts):
    w,beta,pin,norm,dt = p

    dataset_name = "w"+wLabels[ws.index(w)]
    dataset_name += "_beta"+betaLabels[betas.index(beta)]
    if pin is not None:
        dataset_name += "_pin"+str(pin)
    if norm:
        dataset_name += "_instNorm"
    dataset_name += "_dt"+str(dt)

    config = copy.deepcopy(default_dataset_config)
    config['name'] = dataset_name
    config['k'] = None
    config['m'] = None
    config['w0'] = w
    config['beta'] = beta
    config['pin_amplitude'] = pin
    config['instance_norm'] = norm
    config['dt'] = dt

    with open("configs/datasets/"+dataset_name+".yaml","w") as fout:
        yaml.dump(config,fout)

# Training configuration
train_iters = [10_000,20_000,50_000,100_000]
bsizes = [32,64,128,256]
seq_lens = [128,256,512]

with open("configs/train/default_training.yaml","r") as fin:
    default_training_config = yaml.load(fin,Loader=yaml.FullLoader)
for p in product(train_iters,bsizes,seq_lens):
    train_iter,bs,seq_len = p

    training_name = f"iter{train_iter//1000}k"+"_bs"+str(bs)+"_seq"+str(seq_len)

    config = copy.deepcopy(default_training_config)
    config['name'] = training_name
    config['num_train_iters'] = train_iter
    config['bs'] = bs
    config['seq_len'] = seq_len

    with open("configs/train/"+training_name+".yaml","w") as fout:
        yaml.dump(config,fout)


# Optimizer config
learning_rates = [1e-4,5e-4,1e-3]
min_lrs = [1e-5,1e-6,0]
warmups = [True,False]
cos_anneals = [True,False]

with open("configs/opt/default_opt.yaml","r") as fin:
    default_optimizer_config = yaml.load(fin,Loader=yaml.FullLoader)
for p in product(learning_rates,min_lrs,warmups,cos_anneals):
    lr,min_lr,warmup,cos_anneal = p

    optimizer_name = f"lr{lr:.0e}_minlr{min_lr:.0e}"
    if warmup:
        optimizer_name += "_warmup"
    if cos_anneal:
        optimizer_name += "_cosAnneal"

    config = copy.deepcopy(default_optimizer_config)
    config['name'] = optimizer_name
    config['lr'] = lr
    config['min_lr'] = min_lr
    config['warmup'] = warmup
    config['cos_anneal'] = cos_anneal

    with open("configs/opt/"+optimizer_name+".yaml","w") as fout:
        yaml.dump(config,fout)