import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
# from torch.utils.data import DataLoader

# from distshift_dataset import UniqueDistshiftDataset
# from models import AutoencodingWorldModel, SeparatedAutoencodingWorldModel


device = "cuda:0"
to_np = lambda x: x.detach().cpu().numpy()


group_left, group_right = [], []
groups = [(0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)]
for g in groups:
    group_left.append(g[0])
    group_right.append(g[1])
    group_left.append(g[0])
    group_right.append(g[2])
    group_left.append(g[1])
    group_right.append(g[2])


def preprocess(s, a, ns):
    s = (s.to(device) / 255) - .5
    a = a.to(device).float()
    r = r.to(device).float().unsqueeze(1)
    t = t.to(device).float().unsqueeze(1)
    ns = (ns.to(device) / 255) - .5
    return s, a, ns


def train(model, dataloader, n_minibatches, log_interval=25):
    def plot(ax, loss_data, title):
        ax.plot(range(0, n_minibatches, log_interval), loss_data)
        ax.set_title(title)

    step = 0
    losses = []
    image_losses = []
    agent_pos_accs = []
    image_accs = []
    while True:
        for data in dataloader:
            step += 1
            eval = True if step % log_interval == 0 else False

            s, a, ns = data
            s, a, ns = preprocess(s, a, ns)
            results = model.training_step(s, a, ns, eval=eval)

            if step % log_interval == 0:
                print(f"[{step}/{n_minibatches}] loss: {results['loss']:.4f}, acc: {results['image_acc']} , agent pos acc: {results['agent_pos_acc']}")
                losses.append(results['loss'])
                image_losses.append(results['state_reconstruction_loss'])
                agent_pos_accs.append(results["agent_pos_acc"])
                image_accs.append(results['image_acc'])

            if step >= n_minibatches:
                fig, axes = plt.subplots(4, 2)
                plot(axes[0, 0], losses, "Loss")
                plot(axes[0, 1], agent_pos_accs, "AgentPosAcc")
                plot(axes[1, 0], image_losses, "ImageLoss")
                plot(axes[1, 1], image_accs, "ImageAcc")
                
                plt.show()
                return
            

def collate_dataset(dataset):
    s, a, ns = zip(*dataset)
    s = torch.from_numpy(np.stack(s, axis=0)).float().to(device)
    a = torch.from_numpy(np.stack(a, axis=0)).float().to(device).unsqueeze(1) * 2 -1
    ns = torch.from_numpy(np.stack(ns, axis=0)).float().to(device)
    return s, a, ns    


def train_and_eval(model, train_dataset, test_dataset, n_minibatches, log_interval = 20):
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
    
    train_data = collate_dataset(train_dataset)
    test_data = collate_dataset(test_dataset)
    for _ in range(n_minibatches):        
        step += 1
        eval = True if step % log_interval == 0 else False

        s, a, ns = train_data
        
        if np.random.rand() < .4:
            results = model.training_step(s + torch.randn_like(s)*.1, a, ns, eval=eval)
        else:
            results = model.training_step(s, a, ns, eval=eval)

        if step % log_interval == 0:
            append_metrics(results, eval=False)
            s, a, ns = test_data
            results = model.eval_step(s, a, ns)
            print(f"[{step}/{n_minibatches}] loss: {results['loss']:.4f}, acc: {results['image_acc']} , agent pos acc: {results['agent_pos_acc']}")
            append_metrics(results, eval=True)

        if step >= n_minibatches:
            fig, axes = plt.subplots(2, 2)
            plot(axes[0, 0], losses, "Loss")
            plot(axes[0, 1], agent_pos_accs, "AgentPosAcc")
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


# def get_dataloaders():
#     train_dataset = UniqueDistshiftDataset("distshift-v0")
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)
#     test_dataset = UniqueDistshiftDataset("distshift-v1")
#     test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False, num_workers=2)
#     return train_dataloader, test_dataloader


class AutoencodingWorldModel(nn.Module):
    def __init__(self, lr, feature_size=8, alpha=.1, weight_decay=0.) -> None:
        super().__init__()
        self.alpha = alpha
        self.state_encoder = nn.Sequential(
            nn.Conv2d(3, feature_size, kernel_size=(3, 5)),
            nn.SiLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=(3, 3)),
            nn.SiLU(),
        )
        
        self.state_encoder2 = nn.Sequential(
            nn.Conv2d(3, feature_size, kernel_size=(3, 5)),
            nn.SiLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=(3, 3)),
            nn.SiLU(),
        )
        
        self.transition_model = nn.Sequential(
            nn.Linear(feature_size + 1, feature_size*2),
            nn.SiLU(),
            nn.Linear(feature_size*2, feature_size*2),
            nn.SiLU(),
            nn.Linear(feature_size*2, feature_size),
            nn.SiLU(),
        )
        
        # self.state_decoder = nn.Sequential(
        #     nn.ConvTranspose2d(feature_size, feature_size, kernel_size=(3, 3)),
        #     nn.SiLU(),
        #     nn.ConvTranspose2d(feature_size, 3, kernel_size=(3, 5)),
        # )
        self.state_decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_size * 2, feature_size, kernel_size=(3, 5)),
            nn.SiLU(),
            nn.ConvTranspose2d(feature_size, 3, kernel_size=(3, 3)),
        )
        
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.model_name = f"minienc_f_{feature_size}"
        
    def forward(self, state, action):
        state_encoding = self.get_state_encoding(state)
        
        recon_state_encoding = self.state_encoder2(state)
        recon_state_encoding = recon_state_encoding.flatten(1)
        
        state_action = torch.cat((state_encoding, action), dim=1)        
        predicted_next_state_encoding = self.transition_model(state_action)
        
        predictions = dict()
        predictions["next_state_encoding"] = predicted_next_state_encoding
        
        
        predictions["next_state"] = self.state_decoder(torch.cat((predicted_next_state_encoding, recon_state_encoding), axis=1).unsqueeze(-1).unsqueeze(-1))
        predictions["state"] = self.state_decoder(torch.cat((torch.zeros_like(predicted_next_state_encoding), recon_state_encoding), axis=1).unsqueeze(-1).unsqueeze(-1))
        
        
        
        # predictions["state_encoding"] = state_encoding
        # predictions["state"] = self.state_decoder(state_encoding.unsqueeze(-1).unsqueeze(-1))

        return predictions
    
    def get_state_encoding(self, state):
        B, C, H, W = state.shape
        state_encoding = self.state_encoder(state)
        return state_encoding.flatten(1)
    
    def training_step(self, state, action, next_state, eval=False):
        B = state.shape[0]
        predictions = self(state, action)
        
        next_state_reconstruction_loss = nn.functional.mse_loss(predictions["next_state"], next_state)
        state_reconstruction_loss = nn.functional.mse_loss(predictions["state"], state)
        
        loss = next_state_reconstruction_loss + state_reconstruction_loss
        
        locality_loss = torch.tensor(0)
        if not eval:
            locality_loss = nn.functional.mse_loss(predictions["next_state_encoding"][group_left], predictions["next_state_encoding"][group_right])
            loss += self.alpha * locality_loss
            
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            self.optim.step()

        results = {"loss": loss.item(), 
                   "next_state_reconstruction_loss": next_state_reconstruction_loss.item(),
                   "state_reconstruction_loss": state_reconstruction_loss.item(),
                   "locality_loss": locality_loss.item(),}
        if eval:
            results["agent_pos_acc"] = self.agent_pos_acc(predictions["next_state"], next_state)
            results["image_acc"] = self.obs_acc(predictions["next_state"], next_state)
            
        return results

    @torch.no_grad()
    def eval_step(self, s, a, ns):
        return self.training_step(s, a, ns, eval=True)

    @torch.no_grad()
    def obs_acc(self, pred_obs, true_obs, threshold=30):
        def pp(data):
            return (data + .5) * 255
        B, C, H, W = pred_obs.shape
        obs_diff = torch.abs(pp(pred_obs) - pp(true_obs))
        obs_diff = obs_diff.sum(axis=1) < threshold
        return torch.mean(obs_diff.float()).item()
    
    @torch.no_grad()
    def agent_pos_acc(self, pred_obs, true_obs, threshold=20):
        def pp(data):
            rgb = (data + .5) * 255
            return torch.clamp(rgb[:, 2] - (rgb[:, 0] + rgb[:, 1]), 0, 255)
        B, C, H, W = pred_obs.shape
        obs_diff = torch.abs(pp(pred_obs) - pp(true_obs))
        obs_diff = torch.clamp((obs_diff < threshold).sum(axis=(1, 2)) - H*W + 1, 0, 1)
        return torch.mean(obs_diff.float()).item()


def get_dataset():
    train_dataset = []
    test_dataset = []
    base_layer = [[0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 0],
     ]
    base_layer = np.stack([base_layer, base_layer, base_layer], axis=0)
    agent_positions = [(1, 1), (1, 2), (1, 4), (1, 5), (3, 1), (3, 2), (3, 4), (3, 5)]
    
    # fig, axes = plt.subplots(len(agent_positions), 4)
    for i, pos in enumerate(agent_positions):
        state = base_layer.copy()
        state[:, pos[0], pos[1]] = [0, 0, 1]
        for action in [0, 1]:
            if action == 0:
                if pos[1] == 2 or pos[1] == 5:
                    next_state = base_layer.copy()
                    next_state[:, pos[0], pos[1]-1] = [0, 0, 1]
                else:
                    next_state = state.copy()
            elif action == 1:
                if pos[1] == 1 or pos[1] == 4:
                    next_state = base_layer.copy()
                    next_state[:, pos[0], pos[1]+1] = [0, 0, 1]
                else:
                    next_state = state.copy()
            
            # axes[i, action*2].imshow(state.transpose((1, 2, 0)) * 255)
            # axes[i, action*2+1].imshow(next_state.transpose((1, 2, 0)) * 255)
            if pos == (3, 4) or pos == (3, 5):
                test_dataset.append((state, action, next_state))
            else:
                train_dataset.append((state, action, next_state))
    return train_dataset, test_dataset


if __name__=="__main__":
    seed = 42
    import random
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    train_dataset, test_dataset = get_dataset()
    model = AutoencodingWorldModel(lr=.001, alpha=.1, weight_decay=0.0, feature_size=8).to(device)
    train_and_eval(model, train_dataset, test_dataset, n_minibatches=2500*3, log_interval=20)
    # train_dataloader, test_dataloader = get_dataloaders()
    # # train(model, train_dataloader, n_minibatches=5000, log_interval=50)
    # train_and_eval(model, train_dataloader, test_dataloader, n_minibatches=50000, log_interval=100)
    torch.save(model.state_dict(), f"data/models/{model.model_name}.ckpt")


