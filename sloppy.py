import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import copy        

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian

from distshift_dataset import UniqueDistshiftDataset
from models import AutoencodingWorldModel, SeparatedAutoencodingWorldModel
from train import get_dataloaders, device, preprocess
from fim import recover_flattened, load_FIM_inv, flatten_params
# from fim import get_and_save_FIM_data, load_FIM, load_FIM_inv, get_jac_feature_classes


@torch.no_grad()
def eval_model(world_model, dataloader):
    s, a, r, t, ns = next(iter(dataloader))
    s, a, r, t, ns = preprocess(s, a, r, t, ns)
    results = world_model.eval_step(s, a, r, t, ns)
    return results["loss"], results["agent_pos_acc"], results["image_acc"] 


def save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices):
    param_norm = float(torch.linalg.norm(flat_params).cpu().numpy())
    state_dict = recover_flattened(flat_params, flat_param_dict["indices"], flat_param_dict["names"], world_model)
    world_model.load_state_dict(state_dict)
    loss, agent_pos_acc, image_acc = eval_model(world_model, dataloader)
    filtered_parametric_uncertainty = [parametric_uncertainty[i] for i in current_indices]
    
    results.append({"n_eff_params": n_eff_params, "param_norm": param_norm, "step": current_step,
                    "param_uncertainty_mean": np.mean(filtered_parametric_uncertainty), 
                    "param_uncertainty_std": np.std(filtered_parametric_uncertainty),
                    "param_uncertainty_median": np.median(filtered_parametric_uncertainty),
                    "log_param_uncertainty_mean": np.mean(np.log10(filtered_parametric_uncertainty)), 
                    "log_param_uncertainty_std": np.std(np.log10(filtered_parametric_uncertainty)),
                    "log_param_uncertainty_median": np.median(np.log10(filtered_parametric_uncertainty)),
        "loss": loss, "agent_pos_acc": agent_pos_acc, "image_acc": image_acc})


@torch.no_grad()
def reduced_sloppy_model_analysis(original_model: AutoencodingWorldModel, dataloader, steps, strategy="all", remove_most_sloppy=True):        
    world_model = copy.deepcopy(original_model)
    FIM_inv = load_FIM_inv(world_model.model_name, strategy)
    parametric_uncertainty = np.sqrt(np.diag(FIM_inv))
    
    flat_param_dict = flatten_params(world_model)
    flat_params = flat_param_dict["params"]
    current_indices = set(range(len(flat_params)))

    results = []
    
    # Full Model
    n_eff_params = len(flat_params)
    current_step = 0
    save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)
    
    # Full Model with tiny weights set to zero
    set_to_zero = torch.arange(len(flat_params)).to(device)[(torch.abs(flat_params) <= 1e-10).squeeze()]
    flat_params[set_to_zero] = 0
    n_initial_zero_params = len(set_to_zero)

    n_eff_params -= n_initial_zero_params
    parametric_uncertainty[set_to_zero.squeeze().cpu().numpy()] = (np.max(parametric_uncertainty)+1e-10 if remove_most_sloppy else 0)
    current_indices = current_indices - set(set_to_zero.squeeze().cpu().numpy())
    current_step = n_initial_zero_params
    save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)
    
    # Reducing model
    sorted_parameter_indices = (np.argsort(parametric_uncertainty) if remove_most_sloppy else np.argsort(-parametric_uncertainty))
        
    for i in range(n_initial_zero_params, len(parametric_uncertainty), steps):
        current_step = i + steps
        set_to_zero = sorted_parameter_indices[i:i+steps]
        parametric_uncertainty[set_to_zero] = 0
        current_indices = current_indices - set(set_to_zero)
        flat_params[torch.from_numpy(set_to_zero).unsqueeze(1).to(device)] = 0
        n_eff_params -= steps
        n_eff_params = max(n_eff_params, 0)
        save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)

    df = pd.DataFrame(results)
    filename = f"data/results/{world_model.model_name}"
    if not remove_most_sloppy:
        filename += "_reverse"
    df.to_pickle(f"{filename}.pickle")
    
    
@torch.no_grad()
def get_reduced_sloppy_model_baseline_analysis(original_model: AutoencodingWorldModel, dataloader, steps, strategy="all", n_trials=10):
    all_dfs = []
    for _ in range(n_trials):
        world_model = copy.deepcopy(original_model)
        FIM_inv = load_FIM_inv(world_model.model_name, strategy)
        parametric_uncertainty = np.sqrt(np.diag(FIM_inv))
        
        flat_param_dict = flatten_params(world_model)
        flat_params = flat_param_dict["params"]
        current_indices = set(range(len(flat_params)))

        results = []

        # Full Model
        n_eff_params = len(flat_params)
        current_step = 0
        save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)
        
        # Full Model with tiny weights set to zero
        set_to_zero = torch.arange(len(flat_params)).to(device)[(torch.abs(flat_params) <= 1e-10).squeeze()]
        flat_params[set_to_zero] = 0
        n_initial_zero_params = len(set_to_zero)

        n_eff_params -= n_initial_zero_params
        parametric_uncertainty[set_to_zero.squeeze().cpu().numpy()] = 0
        current_indices = current_indices - set(set_to_zero.squeeze().cpu().numpy())
        current_step = n_initial_zero_params
        save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)
        
        # Reducing model
        sorted_parameter_indices = np.argsort(parametric_uncertainty)
        np.random.shuffle(sorted_parameter_indices)
            
        for i in range(n_initial_zero_params, len(parametric_uncertainty), steps):
            current_step = i + steps
            set_to_zero = sorted_parameter_indices[i:i+steps]
            parametric_uncertainty[set_to_zero] = 0
            current_indices = current_indices - set(set_to_zero)
            flat_params[torch.from_numpy(set_to_zero).unsqueeze(1).to(device)] = 0
            n_eff_params -= steps
            n_eff_params = max(n_eff_params, 0)
            save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)

        all_dfs.append(pd.DataFrame(results))
        
    df = (pd.concat(all_dfs)).groupby("n_eff_params", as_index=False).agg('mean')
    filename = f"data/results/{world_model.model_name}"
    df.to_pickle(f"{filename}_random.pickle")
    
    
@torch.no_grad()
def get_reduced_sloppy_model_lowest_highest_analysis(original_model: AutoencodingWorldModel, dataloader, steps, strategy="all"):
        
    for lowest in [True, False]:
        world_model = copy.deepcopy(original_model)
        FIM_inv = load_FIM_inv(world_model.model_name, strategy)
        parametric_uncertainty = np.sqrt(np.diag(FIM_inv))
        
        flat_param_dict = flatten_params(world_model)
        flat_params = flat_param_dict["params"]
        current_indices = set(range(len(flat_params)))

        results = []
        
        # Full Model
        n_eff_params = len(flat_params)
        current_step = 0
        save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)
        
        # Full Model with tiny weights set to zero
        abs_flat_params = torch.abs(flat_params)
        set_to_zero = torch.arange(len(flat_params)).to(device)[(torch.abs(flat_params) <= 1e-10).squeeze()]
        flat_params[set_to_zero] = 0
        n_initial_zero_params = len(set_to_zero)
        abs_flat_params = abs_flat_params.squeeze().cpu().detach().numpy()

        n_eff_params -= n_initial_zero_params
        parametric_uncertainty[set_to_zero.squeeze().cpu().numpy()] = (np.max(abs_flat_params)+1e-10 if not lowest else 0)
        current_indices = current_indices - set(set_to_zero.squeeze().cpu().numpy())
        current_step = n_initial_zero_params
        save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)
        
        # Reducing model
        sorted_parameter_indices = (np.argsort(abs_flat_params) if lowest else np.argsort(-abs_flat_params))
            
        for i in range(n_initial_zero_params, len(parametric_uncertainty), steps):
            current_step = i + steps
            set_to_zero = sorted_parameter_indices[i:i+steps]
            parametric_uncertainty[set_to_zero] = 0
            current_indices = current_indices - set(set_to_zero)
            flat_params[torch.from_numpy(set_to_zero).unsqueeze(1).to(device)] = 0
            n_eff_params -= steps
            n_eff_params = max(n_eff_params, 0)
            save_results(results, world_model, dataloader, flat_params, n_eff_params, current_step, parametric_uncertainty, flat_param_dict, current_indices)

        df = pd.DataFrame(results)
        filename = f"data/results/{world_model.model_name}"
        if lowest:
            filename += "_lowest"
        else:
            filename += "_highest"
        df.to_pickle(f"{filename}.pickle")