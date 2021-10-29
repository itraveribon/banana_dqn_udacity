import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.feature_layer = nn.Sequential(nn.Linear(state_size, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, 16),
                                           nn.ReLU())

        self.state_value_layer = nn.Linear(16, 1)
        self.advantage_layer = nn.Linear(16, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        feature_values = self.feature_layer(state)

        a_values = self.advantage_layer(feature_values)
        s_value = self.state_value_layer(feature_values)

        q_values = s_value + (a_values - a_values.mean())

        return q_values
