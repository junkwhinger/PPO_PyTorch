import os
import numpy as np
import argparse
import yaml
import torch
from hyperopt import fmin, tpe, space_eval, hp, Trials

from agents.ppoagent_search import PPO
from utils.stuff import prepare_dir


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--base_config", type=str, default="search_ppo.yaml")
    args = parser.parse_args()

    with open("config/" + args.base_config) as file:
        config = yaml.full_load(file)

    # search space
    """
    observation_normalization: True
    reward_clipping: True
    value_clipping: True
    reward_standardization: True
    orthogonal_initialization_and_layer_scaling: True
    clipping_gradient: True
    policy_noclip: False
    """
    space = {
        'observation_normalization': hp.choice('observation_normalization', [True, False]),
        'reward_clipping': hp.choice('reward_clipping', [True, False]),
        'value_clipping': hp.choice('value_clipping', [True, False]),
        'reward_standardization': hp.choice('reward_standardization', [True, False]),
        'orthogonal_initialization_and_layer_scaling': hp.choice('orthogonal_initialization_and_layer_scaling', [True, False]),
        'clipping_gradient': hp.choice('clipping_gradient', [True, False]),
        'policy_noclip': hp.choice('policy_noclip', [True, False])
    }

    def objective(params):
        config['seed'] = np.random.randint(78, 1000)
        config.update({"experiment" : params})
        agent = PPO(config)
        best_score = agent.train()
        loss = -best_score
        return loss

    trials = Trials()
    best = fmin(objective, space, trials=trials, algo=tpe.suggest, max_evals=100)
    hyperparams = space_eval(space, best)
    torch.save({"best":hyperparams, "trial":trials, "space": space}, 'search_result.pth')

    print("search done")


if __name__ == "__main__":
    main()