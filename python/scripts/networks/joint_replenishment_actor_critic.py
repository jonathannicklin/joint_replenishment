from typing import Dict

import torch
from torch.nn import Linear, ReLU, Sequential, LayerNorm


class ActorMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, min_val=torch.finfo(torch.float).min, activation=ReLU):
        super(ActorMLP, self).__init__()
        self.min_val = min_val
        self.output_dim = output_dim

        # Create the layers dynamically based on n_layers
        layers = []
        layers.append(Linear(input_dim, hidden_dim))
        layers.append(LayerNorm(hidden_dim))
        layers.append(activation())

        # Add the hidden layers
        for _ in range(n_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim))
            layers.append(LayerNorm(hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(Linear(hidden_dim, output_dim))

        # Stack all layers into a Sequential module
        self.actor = Sequential(*layers)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:

        x = observations['obs']
        x = self.actor(x)

        # If we are in inference mode, mask is optional
        if observations.get('mask') is not None:
            action_masks = observations['mask']
            x[~action_masks] = self.min_val
            x = self.softmax(x)

        return x


class CriticMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, min_val=torch.finfo(torch.float).min, activation=ReLU):
        super(CriticMLP, self).__init__()
        self.min_val = min_val

        # Create the layers dynamically based on n_layers
        layers = []
        layers.append(Linear(input_dim, hidden_dim))
        layers.append(LayerNorm(hidden_dim))
        layers.append(activation())

        # Add the hidden layers
        for _ in range(n_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim))
            layers.append(LayerNorm(hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(Linear(hidden_dim, 1))

        # Stack all layers into a Sequential module
        self.critic = Sequential(*layers)

    def forward(self, observations, state=None, info={}):
        action_masks = observations['mask']
        batch_data = observations['obs']

        x = torch.tensor(batch_data, dtype=torch.float)
        x = self.critic(x)
        return x
