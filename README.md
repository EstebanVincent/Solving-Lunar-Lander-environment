# Solving-Lunar-Lander-environment

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

See notebook for commands to run

## Papers used for this project:

-   [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
-   [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537)

University of Rome, La Sapienza. Artificial Intelligence and Robotics. Reinforcement Learning Course A.Y. 2022/23

Esteban Vincent | Aurélien Lurois
