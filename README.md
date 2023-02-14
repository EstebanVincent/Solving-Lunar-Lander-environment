# Solving-Lunar-Lander-environment

This code defines two classes, ActorNetwork and CriticNetwork, that are used to implement the Proximal Policy Optimization (PPO) actor-critic algorithm in reinforcement learning. The actor network is responsible for deciding the action to take in a given state, while the critic network is responsible for evaluating the state-action value. [1]<br>
A third class DynamicsIdNetwork is used to to predict the dynamics parameters of the env based on the past states and actions of the system. [2]<br>
The ActorNetwork and CriticNetwork are first trained using the default env. <br>
The weights are then transfered for the randomization training. <br>

For one episode :

-   The environment is randomised:
    -   gravity -> integer in range (-11,0)
    -   wind_power -> integer in range (1,20)
    -   turbulance_power -> float(1 decimal) in range (0.1, 2)
-   First 10 steps, the agent is given the action do nothing -> It is in freefall.
    The states are saved and passed to the identification network.
-   The following steps pass the state and the identified dynamics parameters
    to the PPO actor-critic algorithm.

## Folder architecture

-   logs : the logs generated using TensorBoard when training the different models
-   model : the models corresponding to the log
-   ppo : the files creation de PPO
    -   utils : general functions
    -   structures : custom classes for the networks
    -   networks :
        -   ActorNetwork : the actor network of the PPO
        -   CriticNetwork : the critic network of the PPO
        -   DynamicsIdNetwork : the network to reconize dynamic parameters

## Files

-   train.py : The implementation of the PPO actor-critic style and training of the recognition network
-   eval.py : The evaluation of the different models
-   main.py : Access to the 2 previous file via an argparser
-   results.ipynb : Easy to use commands and the results optained visualised.

## How to run

To run the last trained model

```
python main.py --evaluate --d_version 1 --m_version 1 --render
```

See Notebook for more commands to run

## Papers used for this project:

-   [1] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
-   [2] [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537)

University of Rome, La Sapienza. Artificial Intelligence and Robotics. Reinforcement Learning Course A.Y. 2022/23

Esteban Vincent | Aurélien Lurois
