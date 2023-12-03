import numpy as np
import matplotlib.pyplot as plt
import scipy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian

from distshift_dataset import UniqueDistshiftDataset
from models import AutoencodingWorldModel, SeparatedAutoencodingWorldModel
from train import get_dataloaders, device


def get_jac(model, dataloader, strategy="pixels"):
    batched_states, batched_actions, _, _, _ = next(iter(dataloader))
    batched_states = batched_states.to(device).float()
    batched_actions = batched_actions.to(device).float()
    J = []
    for state, action in zip(batched_states, batched_actions):
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        predictions = model(state, action)
        
        if strategy == "pixels":           
            pixels = torch.mean(predictions["next_state"][0], dim=0)
            for row in pixels:
                for pixel in row:
                    model.zero_grad()
                    pixel.backward(retain_graph=True)
                    J.append([param.grad.clone().detach() for param in model.parameters()])
        else:
            for row in predictions["next_state"][0]:
                for pixel in row:
                    for color in pixel:
                        model.zero_grad()
                        color.backward(retain_graph=True)
                        J.append([param.grad.clone().detach() for param in model.parameters()])
    return J


def flatten_jac(J):
    new_J = []
    for J_i in J:
        new_J_i = []
        for grad in J_i:
            new_J_i.append(torch.flatten(grad))
        new_J_i = torch.cat((new_J_i))
        new_J.append(new_J_i)
    return torch.stack(new_J).cpu().detach().numpy()


def get_FIM(world_model, dataloader, strategy="pixel"):
    J = get_jac(world_model, dataloader, strategy)
    return J.T @ J


def get_FIM_inv(world_model, dataloader, strategy="pixel"):
    FIM = get_FIM(world_model, dataloader, strategy)
    return scipy.linalg.pinvh(FIM, atol=1e-8)


def save_FIM(filename, FIM):
    np.savez_compressed(f"data/fims/{filename}", FIM)
    

def save_FIM_inv(filename, FIM_inv):
    np.savez_compressed(f"data/fims_inv/{filename}", FIM_inv)


def load_FIM(filename):
    return np.load(f"data/fims/{filename}.npz")


def load_FIM_inv(filename):
    return np.load(f"data/fims_inv/{filename}.npz")


def get_and_save_FIM_data(world_model, dataloader, strategy="pixel"):
    FIM = get_FIM(world_model, dataloader, strategy)
    filename = f"{world_model.model_name}_{strategy}"
    save_FIM(filename, FIM)
    FIM_inv = scipy.linalg.pinvh(FIM, atol=1e-8)
    save_FIM_inv(filename, FIM_inv)


def main():
    world_model = AutoencodingWorldModel(.001, hidden_layers=[16, 16, 32])
    world_model.load_state_dict(torch.load(f"data/models/autoencoding_161632.ckpt"))
    world_model = world_model.to(device)

    train_dataset = UniqueDistshiftDataset("distshift-v0")
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    get_and_save_FIM_data(world_model, train_dataloader)


if __name__=="__main__":
    main()
