# Banana collection solved with Deep Q Networks
This repository offers you different implementations of Deep Q Networks and test them solving a modified version of the banana collector environment from the Unity ML-Agents toolkit.


The environment consists of an agent navigating along a square world with bananas on the floor that can be collected. 
There are two types of bananas: yellow and blue ones. Collecting a yellow banana returns a reward of +1 while
collecting a blue banana provides a reward of -1. Thus, the objective is to collect as many yellow as possible
bananas while avoiding the blue ones.

The feature or state space has 37 dimensions containing the velocity of the agent and ray-based perception of surrounding objects.
There are four possible actions:

* <kbd>0</kbd> - move forward
* <kbd>1</kbd> - move backward
* <kbd>2</kbd> - turn left
* <kbd>3</kbd> - turn right

The environment is considered to be solved after obtaining an average score greater than 13 over 100 episodes.


# Installation

## 1: Install project dependencies
Please install the pip dependencies with the following command:

<code>pip install -r requirements.txt</code>

## 2: Download the Banana Unity Environment
Download the version corresponding to your operative system and place the uncompressed content in the root path of this
repository:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

# Running the code
We prepared an evaluation script <code>evaluation.py</code> which trains and produced the weights of five models:
* **DQN Network**: The most simplest implementation of a DQN with fixed targets.
* **Double DQN**: Based on the previous one, we implemented a double DQN in order to increase the learning velocity.
* **Dueling DQN**: Based on the simple DQN, we separated the estimation of the state values and action-advantages according 
to the dueling DQN definition.
* **Double Dueling DQN**: This implementation combines the advantages of the two previous ones.
* **Prioritized Replay Experience DQN**: A simple DQN with prioritized replay experience instead of random replay.

Executing the evaluation script will generate one folder under <code>data</data> for each of the above named models. 
Additionally, a PDF named <code>dqn_scores.pdf</code> containing a plot of the scores achieved by each model will be 
saved under the root of this repository.