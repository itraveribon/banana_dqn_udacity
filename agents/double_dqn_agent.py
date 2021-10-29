from agents import dqn_agent

import torch.nn.functional as F


class DoubleAgent(dqn_agent.Agent):
    """Interacts with and learns from the environment."""

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = experiences

        best_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        q_values_target = self.qnetwork_target(next_states).detach()
        q_expected = rewards + (gamma * q_values_target.gather(1, best_actions)) * (1 - dones)
        q_current = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_current)

        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, dqn_agent.TAU)

