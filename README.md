# PPO Implementation in Pytorch for LunarLander-v2
Being fastinated by "IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO",
I wrote PPO code in PyTorch to see if the code-level optimizations work for LunarLander-v2.
And they do! for some extent.

## How to train
Find a config .yaml file in the config directory and run the following command.
You can make your own .yaml file, but make sure they have all the necessary options.
```
$ python main.py --config PPO_M.yaml
```

## How to play
```
$ python main.py --config PPO_M.yaml --eval
```

## How to run Bayesian Optimization for hyperparameters
```
$ python search.py
```

## Reference
- IMPLEMENTATION MATTERS IN DEEP POLICY GRADIENTS: A CASE STUDY ON PPO AND TRPO
- https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
- https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_12/chapter-12.ipynb
- https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/ppo_gae.py
- https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
- https://github.com/openai/baselines/tree/master/baselines/ppo2
- https://github.com/hill-a/stable-baselines/tree/master/stable_baselines/ppo1
- https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12