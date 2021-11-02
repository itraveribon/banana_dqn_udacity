import argparse
import os.path
import shutil

from unityagents import UnityEnvironment
from agents.double_dqn_agent import DoubleAgent
from agents.double_dueling_dqn_agent import DoubleDuelingAgent
from agents.dqn_agent import Agent
from agents.dueling_dqn_agent import DuelingAgent
from agents.prioritized_dqn_agent import PrioritizedAgent

import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def dqn_training(env, n_episodes=1800, eps_start=1.0, eps_end=0.01, eps_decay=0.995, agent_class=Agent):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    agent = agent_class(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size,
                        seed=0)
    solved = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, eps).astype(int)
            env_info = env.step(action)
            env_info = env_info[brain_name]

            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if not solved and np.mean(scores_window) >= 13:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
            solved = True

    model_path = os.path.join(os.path.dirname(__file__), "data", agent.__class__.__name__.lower())
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path)

    agent.save_weights(os.path.join(model_path, "checkpoint.pth"))
    return scores


def evaluation(environment_path: str):
    """Evaluates five reinforcement learning models

    Args:
        environment_path: Path to the banana executable

    Returns:

    """
    env = UnityEnvironment(file_name=os.path.join(os.path.dirname(__file__), environment_path), no_graphics=True)

    scores_dqn = dqn_training(env)

    scores_double_dqn = dqn_training(env, agent_class=DoubleAgent)

    scores_dueling_dqn = dqn_training(env, agent_class=DuelingAgent)

    scores_double_dueling_dqn = dqn_training(env, agent_class=DoubleDuelingAgent)

    scores_prioritized_replay = dqn_training(env, agent_class=PrioritizedAgent)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores_dqn)), scores_dqn, "r-", label="dqn")
    plt.plot(np.arange(len(scores_double_dqn)), scores_double_dqn, "b-", label="double dqn")
    plt.plot(np.arange(len(scores_dueling_dqn)), scores_dueling_dqn, "g-", label="dueling dqn")
    plt.plot(np.arange(len(scores_double_dueling_dqn)), scores_double_dueling_dqn, "y-", label="double dueling dqn")
    plt.plot(np.arange(len(scores_prioritized_replay)), scores_prioritized_replay, "k-", label="prioritized replay dqn")
    plt.legend(loc="upper left")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("dqn_scores.pdf")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--banana_executable", help="Path to the banana environment executable")
    args = parser.parse_args()
    evaluation(args.banana_executable)