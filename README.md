# Solving-Lunar-Lander-environment

## How to run

```
python main.py --train [version]
```

Run the training and save the models in the files:

-   model/actor_model_v[version].pkl
-   model/critic_model_v[version].pkl

The TensorBoard log is in the logs folder

```
python main.py --evaluate [version] --render
```

Run the evaluation for the models in the files:

-   model/actor_model_v[version].pkl
-   model/critic_model_v[version].pkl

The flag render is here to render the evaluation.

## Papers used for this project:

-   [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

University of Rome, La Sapienza. Artificial Intelligence and Robotics. Reinforcement Learning Course A.Y. 2022/23

Esteban Vincent | Aurélien Lurois
