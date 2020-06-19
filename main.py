import os
import argparse
import yaml

from agents.ppoagent import PPO
from utils.stuff import prepare_dir

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="ppo_tanh.yaml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_traj", action="store_true")
    args = parser.parse_args()

    with open("config/" + args.config) as file:
        config = yaml.full_load(file)

    agent_class = globals()[config['agent']]
    agent = agent_class(config)

    if args.eval:
        """play mode"""
        agent.play(num_episodes=args.eval_episodes, save_traj=args.save_traj,
                   record=args.record, save_result=args.save_result, seed=args.seed)
    else:
        print("Training {}...".format(config['exp_name']))
        prepare_dir(config['exp_name'], overwrite=args.overwrite)
        agent.train()
        print("Done\n")

if __name__ == "__main__":
    main()