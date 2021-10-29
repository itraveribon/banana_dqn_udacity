from torch import optim

from agents import (
    double_dqn_agent,
    dqn_agent,
)

from models.dueling_dqn_model import DuelingQNetwork


class DoubleDuelingAgent(double_dqn_agent.DoubleAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(state_size, action_size, seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(dqn_agent.device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(dqn_agent.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=dqn_agent.LR)
