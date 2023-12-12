import numpy as np
import matplotlib.pyplot as plt
import scipy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian

from distshift_dataset import UniqueDistshiftDataset
from models import AutoencodingWorldModel, SeparatedAutoencodingWorldModel
from train import get_dataloaders, device, preprocess


def get_jac(model, dataloader, strategy="pixels"):
    model = model.eval()
    batched_states, batched_actions, _, _, _ = next(iter(dataloader))
    batched_states = batched_states.to(device).float()
    batched_states = (batched_states / 255) - .5
    batched_actions = batched_actions.to(device).float()
    J = []
    for state, action in zip(batched_states, batched_actions):
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        predictions = model(state, action)
        
        if strategy == "pixels":           
            pixels = torch.mean(predictions["next_state"][0], dim=0)
            pixels = torch.flatten(pixels)
            for pixel in pixels:
                model.zero_grad()
                pixel.backward(retain_graph=True)
                J.append([param.grad.clone().detach() for param in model.parameters()])
        elif strategy == "all":
            rgb_pixels = torch.flatten(predictions["next_state"][0])
            for color in rgb_pixels:
                model.zero_grad()
                color.backward(retain_graph=True)
                J.append([param.grad.clone().detach() for param in model.parameters()])
        else: 
            raise ValueError(f"Unrecognized strategy: {strategy}")
    return J


def get_jac_feature_classes(model, dataloader):
    def get_position_to_name(state):
        position_to_name = dict()        
        for r in range(7):
            for c in range(7):
                for color, name in zip(feature_colors, feature_names):
                    if np.all(state[:, r, c] == color):
                        position_to_name[(r, c)] = name
        return position_to_name
                
        
    model = model.eval()
    batched_states, batched_actions, _, _, _ = next(iter(dataloader))
    batched_states = batched_states.to(device).float()
    batched_actions = batched_actions.to(device).float()
    
    feature_names = ["lava", "wall", "floor", "goal", "player"]
    feature_colors = [[255, 0, 0], [0, 0, 0], [255, 255, 255], [0, 255, 0], [0, 0, 255]] 
    feature_J = {name: [] for name in feature_names}
        
    for state, action in zip(batched_states, batched_actions):
        state = state.unsqueeze(0)
        state_np = state[0].detach().cpu().numpy()
        action = action.unsqueeze(0)

        position_to_name = get_position_to_name(state_np)
        
        state = (state / 255) - .5
        predictions = model(state, action)
        
        for r in range(7):
            for c in range(7):
                feature_name = position_to_name[(r, c)]
                for color in predictions["next_state"][0, :, r, c]:
                    model.zero_grad()
                    color.backward(retain_graph=True)
                    feature_J[feature_name].append([param.grad.clone().detach() for param in model.parameters()])
    return feature_J
                    

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
    J = flatten_jac(J)
    return J.T @ J


def get_FIM_inv(world_model, dataloader, strategy="pixel"):
    FIM = get_FIM(world_model, dataloader, strategy)
    return scipy.linalg.pinvh(FIM, atol=1e-8)


def save_FIM(filename, FIM):
    np.savez_compressed(f"data/fims/{filename}", FIM)
    

def save_FIM_inv(filename, FIM_inv):
    np.savez_compressed(f"data/fims_inv/{filename}", FIM_inv)


def load_FIM(filename, strategy="pixel"):
    return np.load(f"data/fims/{filename}_{strategy}.npz")["arr_0"]


def load_FIM_inv(filename, strategy="pixel"):
    return np.load(f"data/fims_inv/{filename}_{strategy}.npz")["arr_0"]


def get_and_save_FIM_data(world_model, dataloader, strategy="pixel"):
    FIM = get_FIM(world_model, dataloader, strategy)
    filename = f"{world_model.model_name}_{strategy}"
    save_FIM(filename, FIM)
    FIM_inv = scipy.linalg.pinvh(FIM, atol=1e-8)
    save_FIM_inv(filename, FIM_inv)
    return FIM, FIM_inv


def flatten_params(model: nn.Module):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    flat_params = []
    names = []
    for (name, param) in model.named_parameters():
        flat_params.append(torch.flatten(param))
        names.append(name)
    # l = [torch.flatten(p) for p in parameters]
    
    indices = []
    s = 0
    for p in flat_params:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat_params = torch.cat(flat_params).view(-1, 1)
    return {"params": flat_params.clone(), "indices": indices, "names": names}


def recover_flattened(flat_params, indices, names, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(model.parameters()):
        l[i] = l[i].view(*p.shape)
    from collections import OrderedDict
    
    state_dict = OrderedDict()
    for n, p in zip(names, l):
        state_dict[n] = p
    return state_dict
    

from models import AutoencodingWorldModel
import pandas as pd
@torch.no_grad()
def reduced_sloppy_model_analysis(world_model: AutoencodingWorldModel, dataloader, steps, strategy="all"):
    def eval_model():
        s, a, r, t, ns = next(iter(dataloader))
        s, a, r, t, ns = preprocess(s, a, r, t, ns)
        results = world_model.eval_step(s, a, r, t, ns)
        return results["loss"], results["agent_pos_acc"], results["image_acc"] 
    
    def save_results():
        param_norm = torch.linalg.norm(flat_params)
        state_dict = recover_flattened(flat_params, flat_param_dict["indices"], flat_param_dict["names"], world_model)
        world_model.load_state_dict(state_dict)
        loss, agent_pos_acc, image_acc = eval_model()
        
        results.append({"n_eff_params": n_eff_params, "param_norm": param_norm, "step": i+steps,
                        "param_uncertainty_mean": np.mean(parameteric_uncertainty), 
                        "param_uncertainty_std": np.std(parameteric_uncertainty),
                        "param_uncertainty_median": np.median(parameteric_uncertainty),
         "loss": loss, "agent_pos_acc": agent_pos_acc, "image_acc": image_acc})
        
        
    FIM_inv = load_FIM_inv(world_model.model_name, strategy)
    parameteric_uncertainty = np.sqrt(np.diag(FIM_inv))
    
    flat_param_dict = flatten_params(world_model)
    flat_params = flat_param_dict["params"]

    results = []
    
    # Full Model
    n_eff_params = len(flat_params)
    save_results()
    
    # Full Model with tiny weights set to zero
    set_to_zero = (torch.abs(flat_params) <= 1e-10)
    flat_params[set_to_zero] = 0
    n_initial_zero_params = torch.sum(set_to_zero)
    n_eff_params -= n_initial_zero_params
    parameteric_uncertainty[set_to_zero] = 0
    save_results()
    
    # Reducing model
    sorted_parameter_indices = np.argsort(parameteric_uncertainty)
    for i in range(n_initial_zero_params, len(parameteric_uncertainty), steps):
        set_to_zero = sorted_parameter_indices[i:i+steps]
        parameteric_uncertainty[set_to_zero] = 0
        flat_params[set_to_zero] = 0
        n_eff_params -= steps
        save_results()

    df = pd.DataFrame(results)
    df.to_pickle(f"data/results/{world_model.model_name}")


def main():
    world_model = AutoencodingWorldModel(.001, hidden_layers=[16, 16, 32])
    world_model.load_state_dict(torch.load(f"data/models/autoencoding_161632.ckpt"))
    world_model = world_model.to(device)

    train_dataset = UniqueDistshiftDataset("distshift-v0")
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    get_and_save_FIM_data(world_model, train_dataloader)


if __name__=="__main__":
    main()
