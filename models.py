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
                 transition_layers=None,
                 weight_decay=0., 
                 conv_dropout_p=.05, 
                 fc_dropout_p=.1, 
                 batch_norm=False, 
                 dropout=False):
        super().__init__()
        self.action_dim = 4
        if hidden_layers is None:
            hidden_layers = [16, 32, 64]
            
        if len(hidden_layers) > 3:
            raise ValueError(f"hidden_layers can have at max 3 elements: Yours contains {len(hidden_layers)}")
                
        if len(hidden_layers) == 3:
            self.obs_enc_dim = hidden_layers[-1]        
        if len(hidden_layers) == 2:
            self.obs_enc_dim = 3*3 * hidden_layers[-1]
        if len(hidden_layers) == 1:
            self.obs_enc_dim = 5*5 * hidden_layers[-1]
        
        if transition_layers is None:
            transition_layers = [hidden_layers[-1], hidden_layers[-1]]
        
        self.state_encoder = self.initialize_state_encoder(hidden_layers, batch_norm, dropout, conv_dropout_p)

        self.transition_model = self.initialize_transition_model(transition_layers, batch_norm, dropout, fc_dropout_p)

        self.state_decoder = self.initialize_state_decoder(hidden_layers, batch_norm, dropout, conv_dropout_p)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.model_name = f"autoencoding_s_{'_'.join(map(str, hidden_layers))}_t_{'_'.join(map(str, transition_layers))}"
        
    def initialize_state_encoder(self, hidden_layers, batch_norm, dropout, conv_dropout_p) -> nn.Module:
        state_encoder = [ConvLayer(3, hidden_layers[0], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p)]
        for i in range(len(hidden_layers) - 1):
            state_encoder.append(ConvLayer(hidden_layers[i], hidden_layers[i+1], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p))
            
        return nn.Sequential(*state_encoder)
    
    def initialize_transition_model(self, transition_layers, batch_norm, dropout, fc_dropout_p) -> nn.Module:
        transition_model = []
        all_transition_layers = [self.obs_enc_dim + self.action_dim] + transition_layers + [self.obs_enc_dim]
        for i in range(len(all_transition_layers)-1):
            if i == (len(all_transition_layers)-2): # if last layer don't do dropout
                transition_model.append(NonLinearLayer(all_transition_layers[i], all_transition_layers[i+1], batch_norm=batch_norm))
            else:
                transition_model.append(NonLinearLayer(all_transition_layers[i], all_transition_layers[i+1], batch_norm=batch_norm, dropout=dropout, dropout_p=fc_dropout_p))
        return nn.Sequential(*transition_model)
    
    def initialize_state_decoder(self, hidden_layers, batch_norm, dropout, conv_dropout_p) -> nn.Module:
        if len(hidden_layers) == 3:
            state_decoder = [
                ConvTransposeLayer(self.obs_enc_dim, hidden_layers[1], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p), 
                ConvTransposeLayer(hidden_layers[1], hidden_layers[0], kernel_size=3, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p), 
                nn.ConvTranspose2d(hidden_layers[0], 3, kernel_size=3), 
                nn.Tanh(),
            ]
        if len(hidden_layers) == 2:
            state_decoder = [
                ConvTransposeLayer(self.obs_enc_dim, hidden_layers[0], kernel_size=5, batch_norm=batch_norm, dropout=dropout, dropout_p=conv_dropout_p), 
                nn.ConvTranspose2d(hidden_layers[0], 3, kernel_size=3), 
                nn.Tanh(),
            ]
        if len(hidden_layers) == 1:
            state_decoder = [
                nn.ConvTranspose2d(self.obs_enc_dim, 3, kernel_size=7), 
                nn.Tanh(),
            ]
            
        return nn.Sequential(*state_decoder, nn.Tanh())
        
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
        return state_encoding.flatten(1)
    
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
    def obs_acc(pred_obs, true_obs, threshold=20):
        def pp(data):
            return (data + .5) * 255
        B, C, H, W = pred_obs.shape
        obs_diff = torch.abs(pp(pred_obs) - pp(true_obs))
        obs_diff = obs_diff.sum(axis=1) < threshold
    
    @torch.no_grad()
    def agent_pos_acc(pred_obs, true_obs, threshold=20):
        def pp(data):
            rgb = (data + .5) * 255
            return torch.clamp(rgb[:, 2] - (rgb[:, 0] + rgb[:, 1]), 0, 255)
        
        B, C, H, W = pred_obs.shape
        obs_diff = torch.abs(pp(pred_obs) - pp(true_obs))
        obs_diff = torch.clamp((obs_diff < threshold).sum(axis=(1, 2)) - H*W + 1, 0, 1)
        return torch.mean(obs_diff.float())


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