import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from distshift_dataset import UniqueDistshiftDataset
from models import AutoencodingWorldModel, SeparatedAutoencodingWorldModel


device = "cuda:0"
to_np = lambda x: x.detach().cpu().numpy()


def preprocess(s, a, r, t, ns):
    s = (s.to(device) / 255) - .5
    a = a.to(device).float()
    r = r.to(device).float().unsqueeze(1)
    t = t.to(device).float().unsqueeze(1)
    ns = (ns.to(device) / 255) - .5
    return s, a, r, t, ns


def train(model, dataloader, n_minitbatches):
    def plot(ax, loss_data, title):
        ax.plot(range(0, n_minitbatches, log_interval), loss_data)
        ax.set_title(title)

    step = 0
    log_interval = 20
    losses = []
    image_losses = []
    image_accs = []
    while True:
        for data in dataloader:
            step += 1
            eval = True if step % log_interval == 0 else False

            s, a, r, t, ns = data
            s, a, r, t, ns = preprocess(s, a, r, t, ns)
            results = model.training_step(s, a, r, t, ns, eval=eval)

            if step % log_interval == 0:
                print(f"[{step}/{n_minitbatches}] loss: {results['loss']}, image_loss: {results['image_loss']}")
                losses.append(results['loss'])
                image_losses.append(results['image_loss'])
                image_accs.append(results['image_acc'])

            if step >= n_minitbatches:
                fig, axes = plt.subplots(4, 2)
                plot(axes[0, 0], losses, "Loss")
                plot(axes[1, 0], image_losses, "ImageLoss")
                plot(axes[1, 1], image_accs, "ImageAcc")
                
                plt.show()
                return
            

def train_and_eval(model, train_dataloader, test_dataloader, n_minibatches, log_interval = 20):
    def plot(ax: plt.Axes, loss_data, title:str):
        if "acc" in title.lower():
            ax.set_ylim(ymin=-.05, ymax=1.05)
        ax.plot(range(0, n_minibatches, log_interval), loss_data[0])
        ax.plot(range(0, n_minibatches, log_interval), loss_data[1])
        ax.set_title(title)

    def append_metrics(results, eval):
        idx = 1 if eval else 0
        losses[idx].append(results['loss'])
        image_losses[idx].append(results['state_reconstruction_loss'])
        agent_pos_accs[idx].append(results["agent_pos_acc"])
        image_accs[idx].append(results['image_acc'])

    step = 0
    losses = [[], []]
    image_losses = [[], []]
    agent_pos_accs = [[], []]
    image_accs = [[], []]
    while True:
        for data in train_dataloader:
            step += 1
            eval = True if step % log_interval == 0 else False

            s, a, r, t, ns = data
            s, a, r, t, ns = preprocess(s, a, r, t, ns)
            results = model.training_step(s, a, r, t, ns, eval=eval)

            if step % log_interval == 0:
                append_metrics(results, eval=False)
                data = next(iter(test_dataloader))
                s, a, r, t, ns = data
                s, a, r, t, ns = preprocess(s, a, r, t, ns)
                results = model.eval_step(s, a, r, t, ns)
                print(f"[{step}/{n_minibatches}] loss: {results['loss']:.4f}, ")
                append_metrics(results, eval=True)

            if step >= n_minibatches:
                fig, axes = plt.subplots(2, 2)
                plot(axes[0, 0], losses, "Loss")
                plot(axes[0, 1], image_accs, "AgentPosAcc")
                plot(axes[1, 0], image_losses, "ImageLoss")
                plot(axes[1, 1], image_accs, "ImageAcc")
                handles, labels = axes[0,0].get_legend_handles_labels()
                fig.legend(handles, labels=["Train", "Val"])
                
                plt.show()
                return


def postprocess(img):
    post_img = (img + .5) * 255
    post_img = post_img.transpose((1, 2, 0))
    return np.clip(post_img, 0, 255).astype(int)


def get_dataloaders():
    train_dataset = UniqueDistshiftDataset("distshift-v0")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)
    test_dataset = UniqueDistshiftDataset("distshift-v1")
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False, num_workers=2)
    return train_dataloader, test_dataloader


if __name__=="__main__":
    model = AutoencodingWorldModel(lr=.001, weight_decay=0.00001, hidden_layers=[32, 64, 64], dropout=True, batch_norm=False).to(device)
    train_dataloader, test_dataloader = get_dataloaders()
    train_and_eval(model, train_dataloader, test_dataloader, n_minibatches=5000, log_interval=25)
    torch.save(model.state_dict(), f"data/models/autoencoding.ckpt")
    