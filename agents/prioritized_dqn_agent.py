import numpy as np
import torch

from agents import dqn_agent
from agents.prioritized_replay_buffer_sumtree import PrioritizedReplayBuffer


class PrioritizedAgent(dqn_agent.Agent):
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

        # Replay memory
        self.memory = PrioritizedReplayBuffer(dqn_agent.BUFFER_SIZE, dqn_agent.BATCH_SIZE, seed, dqn_agent.device)
        self.importance_sampling_coeff = 0.4  # importance-sampling, from initial value increasing to 1

        self.importance_increment_per_sampling = 0.001

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones, indices, probabilities = experiences

        q_expected = rewards + (gamma * self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)) * (1 - dones)
        q_current = self.qnetwork_local(states).gather(1, actions)

        delta = (q_expected - q_current)
        absolute_errors = np.abs(delta.detach())
        self.memory.batch_update(indices, absolute_errors)

        sampling_weights = np.power((1 / len(self.memory)) * (1 / probabilities), self.importance_sampling_coeff)
        loss = torch.mean((delta * sampling_weights)**2)

        self.importance_sampling_coeff += self.importance_increment_per_sampling
        self.importance_sampling_coeff = min(self.importance_sampling_coeff, 1.0)

        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, dqn_agent.TAU)

