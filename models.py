import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, batch_norm=False, dropout=False, dropout_p=.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
        )
        
    def forward(self, x):
        return self.net(x)


class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=False, dropout=False, dropout_p=.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
        )
        
    def forward(self, x):
        return self.net(x)
    

class NonLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, batch_norm=False, dropout=False, dropout_p=.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features) if batch_norm else nn.Identity(), 
            nn.SiLU(),
            nn.Dropout(p=dropout_p) if dropout else nn.Identity(),
        )
        
    def forward(self, x):
        return self.net(x)


class AutoencodingWorldModel(nn.Module):
    def __init__(self, 
                 lr, 
                 hidden_layers=None, 
                 latent_dim=None, 
                 weight_decay=0., 
                 conv_dropout_p=.05, 
                 fc_dropout_p=.1, 
                 batch_norm=False, 
                 dropout=False):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [16, 32, 64]
        if latent_dim is None:
            latent_dim = hidden_layers[-1]
        
        self.latent_dim = latent_dim
        self.obs_enc_dim = hidden_layers[-1]
        
        self.state_encoder = nn.Sequential(
            ConvLayer(3, hidden_layers[0], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p),
            ConvLayer(hidden_layers[0], hidden_layers[1], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p),
            # ConvLayer(hidden_layers[1], hidden_layers[1], kernel_size=3, padding=1, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p),
            ConvLayer(hidden_layers[1], hidden_layers[2], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p),
        )

        self.transition_model = nn.Sequential(NonLinearLayer(self.obs_enc_dim + 4, hidden_layers[-1], batch_norm=batch_norm, dropout=dropout, dropout_p=fc_dropout_p),
                                      NonLinearLayer(hidden_layers[-1], hidden_layers[-1], batch_norm=batch_norm, dropout=dropout, dropout_p=fc_dropout_p),
                                      NonLinearLayer(hidden_layers[-1], self.latent_dim, batch_norm=False, dropout=False))

        self.state_decoder = nn.Sequential(
            ConvTransposeLayer(self.latent_dim, hidden_layers[1], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p), 
            ConvTransposeLayer(hidden_layers[1], hidden_layers[0], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p), 
            # nn.ConvTranspose2d(hidden_layers[0], hidden_layers[0], kernel_size=3, padding=1), 
            # nn.SiLU(),
            nn.ConvTranspose2d(hidden_layers[0], 3, kernel_size=3), 
            nn.Tanh(),
        )
        
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
    def forward(self, state, action):
        state_encoding = self.get_state_encoding(state)
        
        state_action = torch.cat((state_encoding, action), dim=1)        
        predicted_next_state_encoding = self.transition_model(state_action)
        
        predictions = dict()
        predictions["next_state_encoding"] = predicted_next_state_encoding
        predictions["next_state"] = self.state_decoder(predicted_next_state_encoding.unsqueeze(-1).unsqueeze(-1))

        return predictions
    
    def get_state_encoding(self, state):
        B, C, H, W = state.shape
        state_encoding = self.state_encoder(state)
        return state_encoding.squeeze(-1).squeeze(-1)
    
    def training_step(self, state, action, r, t, next_state, eval=False):
        B = state.shape[0]
        predictions = self(state, action)
        
        next_state_reconstruction_loss = nn.functional.mse_loss(predictions["next_state"], next_state)
        
        loss = next_state_reconstruction_loss
        
        if not eval:
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
            self.optim.step()

        results = {"loss": loss.item(), 
                   "state_reconstruction_loss": next_state_reconstruction_loss.item()}
        if eval:
            results["agent_pos_acc"] = self.agent_pos_acc(predictions["next_state"], next_state)
            results["image_acc"] = self.obs_acc(predictions["next_state"], next_state)
            
        return results

    @torch.no_grad()
    def eval_step(self, s, a, r, t, ns):
        return self.training_step(s, a, r, t, ns, eval=True)

    @torch.no_grad()
    def obs_acc(self, pred_obs, true_obs, threshold=20):
        def pp(data):
            return (data + .5) * 255
        B, C, H, W = pred_obs.shape
        obs_diff = torch.abs(pp(pred_obs) - pp(true_obs))
        obs_diff = obs_diff.sum(axis=1)
        return (torch.sum(obs_diff < threshold) / (B * H * W)).item()
    
    @torch.no_grad()
    def agent_pos_acc(self, pred_obs, true_obs, threshold=20):
        def pp(data):
            return (data + .5) * 255
        B, C, H, W = pred_obs.shape
        obs_diff = torch.abs(pp(pred_obs) - pp(true_obs))
        obs_diff = obs_diff[:, 2].sum(axis=(1, 2))
        return (torch.sum(obs_diff < threshold) / B).item()


class SeparatedAutoencodingWorldModel(AutoencodingWorldModel):
    def forward(self, s, a):
        state_encoding = self.get_state_encoding(s)
        next_state_encoding = self.get_state_encoding(s)
        
        state_action = torch.cat((state_encoding.detach(), a), dim=1)  # .detach() so transition model is trained separately from autoencoder
        predicted_next_state_encoding = self.transition_model(state_action)

        predictions = dict()
        predictions["true_next_state_encoding"] = next_state_encoding.detach()
        
        predictions["state_decoding"] = self.state_decoder(state_encoding.unsqueeze(-1).unsqueeze(-1))
        predictions["next_state_decoding"] = self.state_decoder(next_state_encoding.unsqueeze(-1).unsqueeze(-1))
        
        predictions["predicted_next_state_encoding"] = predicted_next_state_encoding
        return predictions

    def training_step(self, state, action, r, t, next_state, eval=False):
        B = state.shape[0]
        predictions = self(state, action)
        
        state_reconstruction_loss = nn.functional.mse_loss(predictions["state_decoding"], state)
        state_reconstruction_loss += nn.functional.mse_loss(predictions["next_state_decoding"], state)
        
        transition_loss = nn.functional.mse_loss(predictions["predicted_next_state_encoding"], predictions["true_next_state_encoding"])
        
        loss = .5 * state_reconstruction_loss + transition_loss
        
        if not eval:
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
            self.optim.step()

        results = {"loss": loss.item(), 
                   "state_reconstruction_loss": state_reconstruction_loss.item(),
                   "transition_loss": transition_loss.item()}
        
        if eval:
            results["agent_pos_acc"] = self.agent_pos_acc(predictions["next_obs"], next_state)
            results["image_acc"] = self.obs_acc(predictions["next_obs"], next_state)

        return results

    @torch.no_grad()
    def eval_step(self, s, a, r, t, ns):
        B = s.shape[0]
        predictions = self(s, a)
        
        next_state_reconstruction_loss = nn.functional.mse_loss(predictions["next_state"], ns)

        loss = next_state_reconstruction_loss

        results = {"loss": loss.item(), 
                   "state_reconstruction_loss": next_state_reconstruction_loss.item()}
        
        results["agent_pos_acc"] = self.agent_pos_acc(predictions["next_state"], ns)
        results["image_acc"] = self.obs_acc(predictions["next_state"], ns)

        return results