import torch
import torch.nn as nn
from model import GPT, GPTConfig
import json

def load_model(name,date,epoch=None):
    src = f"trainings/{name}_{date}/"
    with open(f"{src}/model_cfg.json","r") as fin:
        model_cfg = json.load(fin)
    with open(f"{src}/training_cfg.json","r") as fin:
        train_cfg = json.load(fin)

    dt = train_cfg['dt']
    k = train_cfg['k']
    beta = train_cfg['beta']
    pin_amplitude = train_cfg['pin_amplitude']
    min_amplitude = train_cfg['min_amplitude']
    k_context = train_cfg['k_context']

    # define model
    gpt_config = GPTConfig(**model_cfg)
    model = GPT(gpt_config)
    if epoch is None:
        model_name = f"{name}_{date}_best.pt"
    else:
        model_name = f"{name}_{date}_best_epoch_{epoch}.pt"
    model.load_state_dict(torch.load(f"{src}/{model_name}",map_location='cpu'))
    model.eval()

    kwargs = dict(
        dt=dt,
        k=k,
        beta=beta,
        pin_amplitude=pin_amplitude,
        min_amplitude=min_amplitude,
        k_context=k_context
    )

    return model, kwargs