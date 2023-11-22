import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle 

import torch

from train import WorldModel, get_dataloaders, device, UniqueDistshiftDataset, DataLoader, preprocess, postprocess, to_np


def get_original_test_dataloaders():
    test_dataset = UniqueDistshiftDataset(filenames=["distshift-v0", "distshift-v1"])
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False)
    return test_dataloader


@torch.no_grad()
def get_acc(model, dataloader):
    data = next(iter(dataloader))
    s, a, r, t, ns = data
    s, a, r, t, ns = preprocess(s, a, r, t, ns)
    predictions = model(s, a)

    def obs_acc(pred_obs, true_obs, threshold=20):
        def pp(data):
            return (data + .5) * 255
        B, C, H, W = pred_obs.shape
        obs_diff = pp(pred_obs) - pp(true_obs)
        obs_diff = obs_diff.sum(axis=1)
        return torch.sum(obs_diff < threshold) / (B * H * W)
    
    def termination_acc(pred_t, t):
        B, C = pred_t.shape
        pred_t[pred_t < .5] = 0
        pred_t[pred_t >= .5] = 1
        
        return torch.sum(pred_t == t) / B

    def reward_acc(pred_r, r, threshold=.2):
        B, C = pred_r.shape
        gt_lowlim = (pred_r > (r - threshold))
        lt_highlim = (pred_r < (r + threshold))
        return torch.sum(torch.logical_and(gt_lowlim, lt_highlim)) / B

    o_acc = obs_acc(predictions["next_obs"][:, :, 1:-1, 1:-1], ns[:, :, 1:-1, 1:-1])
    t_acc = termination_acc(predictions["termination"], t)
    r_acc = reward_acc(predictions["reward"], r)
    
    return o_acc.item(), t_acc.item(), r_acc.item()


if __name__=="__main__":
    categories = ["all", "horizontal", "vertical", "len1", "len2"]
    test_dataloaders = dict()
    for category in categories:
        if category == "all":
            continue
        _, test_dataloader = get_dataloaders(category=category, n_seeds=30)
        test_dataloaders[category] = test_dataloader
        
    test_dataloaders["original"] = get_original_test_dataloaders()
    
    model_data = []
    
    with torch.no_grad():
        for category in categories:
            model = WorldModel(lr=.001, weight_decay=0.00001, hidden_layers=[32, 64, 64], dropout=True, batch_norm=False)
            model.load_state_dict(torch.load(f"data/models/distshift_{category}.ckpt"))
            model = model.eval().to(device)
            
            for test_category, test_dataloader in test_dataloaders.items():
                o_acc, t_acc, r_acc = get_acc(model, test_dataloader)
                data = next(iter(test_dataloader))
                s, a, r, t, ns = data
                s, a, r, t, ns = preprocess(s, a, r, t, ns)
                representations = model.get_representations(s, a)
                
                model_data.append({
                    "model": category,
                    "data": test_category,
                    "img_acc": o_acc,
                    "termination_acc": t_acc,
                    "reward_acc": r_acc,
                    "representation": to_np(representations)
                })
                
    with open("data/model_data.pickle", 'wb') as handle:
        pickle.dump(model_data, handle, protocol=pickle.HIGHEST_PROTOCOL)