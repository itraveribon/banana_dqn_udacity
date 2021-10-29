import random
from collections import (
    deque,
    namedtuple,
)

import numpy as np
import torch

from agents.dqn_agent import device
from agents.replay_buffer import ReplayBuffer


class PrioritizedReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = list()
        self.memory_priorities = dict()
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.experience_priority = namedtuple("ExperiencePriority", field_names=["priority", "probability", "weight"])
        self.seed = random.seed(seed)
        self.device = device

        self._saved_experiences = 0
        self._max_priority = 1.0
        self.epsilon = 1e-6

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.memory_priorities[self._saved_experiences] = self.experience_priority(self._max_priority, self._max_priority)
        self._saved_experiences += 1

    def sample(self, dqn_network, gamma):
        """Randomly sample a batch of experiences from memory."""
        experience_states = [experience.state for experience in self.memory]
        experience_actions = [experience.action for experience in self.memory]
        experience_rewards = [experience.reward for experience in self.memory]
        next_states = [experience.next_state for experience in self.memory]
        dones = np.asarray([experience.done for experience in self.memory])

        q_expected = experience_rewards + (gamma * dqn_network(next_states).detach().max(1)[0].unsqueeze(1)) * (1 - dones)
        q_current = dqn_network(experience_states).gather(1, experience_actions)

        errors = np.abs(q_expected - q_current) + self.epsilon

        probabilities = [single_error / sum(errors) for single_error in errors]

        experiences = random.choices(self.memory, weights=probabilities, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)