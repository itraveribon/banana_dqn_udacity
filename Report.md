# Implementation details
We developed five learning agents and therefore models in this repository:
* **DQN Network**: The most simplest implementation of a DQN with fixed targets.
* **Double DQN**: Based on the previous one, we implemented a double DQN in order to increase the learning velocity.
* **Dueling DQN**: Based on the simple DQN, we separated the estimation of the state values and action-advantages according 
to the dueling DQN definition.
* **Double Dueling DQN**: This implementation combines the advantages of the two previous ones.
* **Prioritized Replay Experience DQN**: A simple DQN with prioritized replay experience instead of random replay.

Each one of the agents is able to solve the environment, i.e., to obtain an average reward of +13 over 100 episodes. All
of them were executed with the same set of hyperparameters:

```python
BUFFER_SIZE = int(np.power(2, 17))  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
```

Behind the five above named agents there are two kind of networks described in the <code>models</code> module:
* dqn_model: A neural networks consisting of three linear layers. The amount of neurons for the hidden layers are 
[32, 16, action_size]. Thus, we reduce in the first layer the 37 space dimensions to 32 and continue reducing them until
we reach the dimension of the action space.
* dueling_dqn_model: A neural network consisting of four linear layers and a bifurcation following the dueling schema:
```
                                                              Linear(16, 1)
                                                           /                 \        
                                                          /                   \
                                                         /                     \
                                                        /                       \
                                                       /                         \
---space_size---Linear(space_size, 32)---Linear(32, 16)                            +
                                                        \                         /
                                                         \                       /
                                                          Linear(16, action_size)
```

The `output.log` file contains the output of the evaluation.py script. As you can see, all the implemented agents solved
the environment However, the amount of needed episodes is different for each one. Double Dueling DQN was the fastest one
solving the environment in 406 episodes. However, Prioritized Experience Replay was the agent achieving the highest 
average score. Please, see the table below to check the scores and amount of episodes of each agent:

Agent | #Episodes | Avg. Score | Max Avg. Score
----- | --------- | ---------- | --------------
Simple DQN | 501 | 13.02 | 16.45
Double DQN | 446 | 13.01 | 17.06
Dueling DQN | 449 | 13.07 | 16.79
Double Dueling DQN | 406 | 13.02 | 16.96
Prioritized Experience Replay DQN | 412 | 13.02 | 17.90 

Regardless of the amount of episodes, the Prioritized Experience Replay DQN is the slowest one in terms of seconds. The 
way the experiences are sampled increases the time complexity of the algorithm and therefore the time needed for each 
episode.

![img_1.png](img_1.png)

# Future work
In order to improve the current implementation I would suggest to combine all the improvements we implemented here within
a "Rainbow" agent. 

The implemented neural networks seems to be complex enough to solve the model. I do not expect any 
quick win by increasing the model complexity. An adjustment of hyperparameter could be ever more helpful. 
