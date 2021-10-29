import random
from collections import (
    deque,
    namedtuple,
)

import numpy as np
import torch

from agents.sumtree import SumTree


class PrioritizedReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    MINIMUM_ABS_ERROR = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PRIORITIZATION_EXP = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, buffer_size, batch_size, seed, device):
        self.batch_size = batch_size
        random.seed(seed)
        self.device = device

        # Making the tree
        self.tree = SumTree(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        max_priority = self.tree.max_priority

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        e = self.experience(state, action, reward, next_state, done)
        self.tree.add(max_priority, e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        values_to_search = np.random.uniform(0, self.tree.sum_priorities, self.batch_size)
        picked_indices = []
        experiences = []
        experience_probabilities = []

        for leaf_value in values_to_search:
            index, priority, experience = self.tree.get_leaf(leaf_value)
            if experience is not None:
                picked_indices.append(index)
                experience_probabilities.append(priority / self.tree.sum_priorities)
                experiences.append(experience)

        for e in experiences:
            if e is not None:
                try:
                    state = e.state
                except AttributeError as exception:
                    print(exception)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        probabilities = torch.from_numpy(np.vstack(experience_probabilities)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones, picked_indices, probabilities)

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.MINIMUM_ABS_ERROR  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PRIORITIZATION_EXP)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.tree)
