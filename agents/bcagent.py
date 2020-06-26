import os
import numpy as np
from collections import deque
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from models import Actor, Critic, Discriminator
from utils.replay_buffer import PPOMemory
from utils.stuff import RewardScaler, ObservationScaler
from torch.utils.tensorboard import SummaryWriter
from gail import get_gail_dataset
from behaviour_cloning import get_bc_dataset, pretrain


class BC(BaseAgent):
    def __init__(self, config):
        super(BC, self).__init__()
        self.config = config
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

        self.actor = Actor(device=self.config['device'],
                           input_dim=self.config['env']['nS'], output_dim=self.config['env']['nA'],
                           hidden_dims=self.config['model']['actor']['hidden_dims'],
                           hidden_activation_fn=self.config['model']['actor']['hidden_acivation_fn'],
                           weight_init_scheme="normal")

    def train(self):
        writer_path = os.path.join('experiments', self.config['exp_name'], 'runs')
        self.writer = SummaryWriter(writer_path)

        bc_train_set, bc_valid_set = get_bc_dataset(self.config['train']['bc']['samples_exp_name'],
                                                    self.config['train']['bc']['minimum_score'],
                                                    self.config['train']['bc']['batch_size'],
                                                    self.config['train']['bc']['demo_count'],
                                                    self.config['train']['bc']['val_size'])

        self.actor = pretrain(self.actor,
                              self.config['train']['bc']['lr'],
                              self.config['train']['bc']['epochs'],
                              bc_train_set, bc_valid_set,
                              use_obs_scaler=False, writer=self.writer)

        self.save_weight(self.actor, self.config['exp_name'], "best")

    def play(self, num_episodes=1, save_traj=False, seed=9999, record=False, save_result=False):

        # load policy
        self.load_weight(self.actor, self.config['exp_name'])

        env = self.init_env(self.config['env']['name'])
        env.seed(seed)
        if record:
            from gym import wrappers
            rec_dir = os.path.join("experiments", self.config['exp_name'], "seed_{}".format(seed))
            env = wrappers.Monitor(env, rec_dir, force=True)
        scores, trajectories = [], []

        for episode in range(num_episodes):
            current_trajectory = []
            episode_score = 0

            # initialize env
            state = env.reset()

            while True:
                # env.render()

                # select greedy action
                with torch.no_grad():
                    action_tensor = self.actor.select_greedy_action(state)
                action = action_tensor.numpy()[0]  # single env

                current_trajectory.append((state, action))

                # run action
                next_state, reward, done, _ = env.step(action)

                # add reward
                episode_score += reward

                # update state
                state = next_state

                # game over condition
                if done:
                    scores.append(episode_score)
                    trajectories.append((current_trajectory, episode_score))
                    break

        avg_score = np.mean(scores)
        print("Average score {} on {} games".format(avg_score, num_episodes))
        if save_result:
            played_result_path = os.path.join("experiments", self.config['exp_name'], "runs", "play_score.pth")
            torch.save(scores, played_result_path)

        if save_traj:
            demo_dir = os.path.join("experiments", self.config['exp_name'], "demonstration")
            os.makedirs(demo_dir)
            torch.save(trajectories, os.path.join(demo_dir, "demo.pth"))
            print("saved {} trajectories.".format(num_episodes))

        env.close()
