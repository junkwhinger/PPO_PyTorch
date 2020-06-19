import os
import numpy as np
from collections import deque
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from models import Actor, Critic
from utils.replay_buffer import PPOMemory
from utils.stuff import RewardScaler, ObservationScaler


class PPO(BaseAgent):
    def __init__(self, config):
        super(PPO, self).__init__()
        self.config = config
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

        if self.config['experiment']['orthogonal_initialization_and_layer_scaling']:
            weight_init_scheme = 'orthogonal'
        else:
            weight_init_scheme = 'normal'

        self.actor = Actor(device=self.config['device'],
                           input_dim=self.config['env']['nS'], output_dim=self.config['env']['nA'],
                           hidden_dims=self.config['model']['actor']['hidden_dims'],
                           hidden_activation_fn=self.config['model']['actor']['hidden_acivation_fn'],
                           weight_init_scheme=weight_init_scheme)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=self.config['model']['actor']['lr'],
                                          betas=self.config['model']['actor']['betas'])

        self.critic = Critic(device=self.config['device'],
                             input_dim=self.config['env']['nS'],
                             hidden_dims=self.config['model']['critic']['hidden_dims'],
                             hidden_activation_fn=self.config['model']['critic']['hidden_acivation_fn'],
                             weight_init_scheme=weight_init_scheme)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=self.config['model']['critic']['lr'],
                                           betas=self.config['model']['critic']['betas'])

        # [EXPERIMENT] - reward scaler: r / rs.std()
        if self.config['experiment']['reward_standardization']:
            self.reward_scaler = RewardScaler(gamma=self.config['train']['gamma'])

        # [EXPERIMENT] - observation scaler: (ob - ob.mean()) / (ob.std())
        if self.config['experiment']['observation_normalization']:
            self.observation_scaler = ObservationScaler()


    # train
    def train(self):
        """
        # initialize env, memory
        # foreach episode
        #   foreach timestep
        #     select action
        #     step action
        #     add exp to the memory
        #     if done or timeout or memory_full: update gae & tdlamret
        #     if memory is full
        #       bootstrap value
        #       optimize
        #       clear memory
        #     if done:
        #       wrapup episode
        #       break
        """
        self.best_score = np.float("-inf")

        # prepare env, memory, stuff
        env = self.init_env(self.config['env']['name'])
        env.seed(self.config['seed'])
        self.memory = PPOMemory(gamma=self.config['train']['gamma'],
                           tau=self.config['train']['gae']['tau'])
        score_queue = deque(maxlen=self.config['train']['average_interval'])
        length_queue = deque(maxlen=self.config['train']['average_interval'])

        for episode in trange(1, self.config['train']['max_episodes']+1):
            self.episode = episode
            episode_score = 0

            # reset env
            state = env.reset()

            for t in range(1, self.config['train']['max_steps_per_episode']+1):
                # if self.episode % 100 == 0:
                #     env.render()

                # [EXPERIMENT] - observation scaler: (ob - ob.mean()) / (ob.std())
                if self.config['experiment']['observation_normalization']:
                    state = self.observation_scaler(state, update=True)

                # select action & estimate value from the state
                with torch.no_grad():
                    state_tensor = torch.tensor(state).unsqueeze(0).float() # bsz = 1
                    action_tensor, logpa_tensor = self.actor.select_action(state_tensor)
                    value_tensor = self.critic(state_tensor).squeeze(1) # don't need bsz dim

                # step action
                action = action_tensor.numpy()[0] # single worker
                next_state, reward, done, _ = env.step(action)

                # update episode_score
                episode_score += reward

                # [EXPERIMENT] - reward scaler r / rs.std()
                if self.config['experiment']['reward_standardization']:
                    reward = self.reward_scaler(reward, update=True)

                # [EXPERIMENT] - reward clipping [-5, 5]
                if self.config['experiment']['reward_clipping']:
                    reward = np.clip(reward, -5, 5)

                # add experience to the memory
                self.memory.store(s=state, a=action, r=reward, v=value_tensor.item(), lp=logpa_tensor.item())

                # done or timeout or memory full
                # done => v = 0
                # timeout or memory full => v = critic(next_state)
                # update gae & return in the memory!!
                timeout = t == self.config['train']['max_steps_per_episode']
                time_to_optimize = len(self.memory) == self.config['train']['ppo']['memory_size']
                if done or timeout or time_to_optimize:
                    if done:
                        # cuz the game is over, value of the next state is 0
                        v = 0
                    else:
                        # if not, estimate it with the critic
                        next_state_tensor = torch.tensor(next_state).unsqueeze(0).float() # bsz = 1
                        with torch.no_grad():
                            next_value_tensor = self.critic(next_state_tensor).squeeze(1)
                        v = next_value_tensor.item()

                    # update gae & tdlamret
                    self.memory.finish_path(v)

                # if memory is full, optimize PPO
                if time_to_optimize:
                    self.optimize()

                if done:
                    score_queue.append(episode_score)
                    length_queue.append(t)
                    break

                # update state
                state = next_state

            avg_score = np.mean(score_queue)

            if avg_score >= self.best_score:
                self.best_score = avg_score


        return self.best_score

    # optimize
    def optimize(self):
        data = self.prepare_data(self.memory.get())
        self.optimize_ppo(data)

    def prepare_data(self, data):
        states_tensor = torch.from_numpy(np.stack(data['states'])).float() # bsz, 8
        actions_tensor = torch.tensor(data['actions']).long() # bsz
        logpas_tensor = torch.tensor(data['logpas']).float() # bsz
        tdlamret_tensor = torch.tensor(data['tdlamret']).float() # bsz
        advants_tensor = torch.tensor(data['advants']).float() # bsz
        values_tensor = torch.tensor(data['values']).float() # bsz

        data_tensor = dict(states=states_tensor, actions=actions_tensor, logpas=logpas_tensor,
                    tdlamret=tdlamret_tensor, advants=advants_tensor, values=values_tensor)

        return data_tensor

    def ppo_iter(self, batch_size, ob, ac, oldpas, atarg, tdlamret, vpredbefore):
        total_size = ob.size(0)
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        n_batches = total_size // batch_size
        for nb in range(n_batches):
            ind = indices[batch_size * nb : batch_size * (nb+1)]
            yield ob[ind], ac[ind], oldpas[ind], atarg[ind], tdlamret[ind], vpredbefore[ind]

    def optimize_ppo(self, data):

        """
        https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py line 164

        # get data from the memory
        # prepare dataloader
        # foreach optim_epochs
        #   foreach batch
        #     calculate loss and gradient
        #     update nn
        """

        ob = data['states']
        ac = data['actions']
        oldpas = data['logpas']
        atarg = data['advants']
        tdlamret = data['tdlamret']
        vpredbefore = data['values']

        # normalize advant a.k.a atarg
        atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-5)

        # can't be arsed..
        eps = self.config['train']['ppo']['clip_range']

        policy_losses = []
        entropy_losses = []
        value_losses = []

        # foreach policy_update_epochs
        for i in range(self.config['train']['ppo']['optim_epochs']):
            # foreach batch
            data_loader = self.ppo_iter(self.config['train']['ppo']['batch_size'],
                                        ob, ac, oldpas, atarg, tdlamret, vpredbefore)
            for batch in data_loader:
                ob_b, ac_b, old_logpas_b, atarg_b, vtarg_b, old_vpred_b = batch

                # policy loss
                cur_logpas, cur_entropies = self.actor.get_predictions(ob_b, ac_b)
                ratio = torch.exp(cur_logpas - old_logpas_b)

                # clip ratio
                clipped_ratio = torch.clamp(ratio, 1.-eps, 1.+eps)

                # policy_loss
                surr1 = ratio * atarg_b

                if self.config['experiment']['policy_noclip']:
                    pol_surr = -surr1.mean()
                else:
                    surr2 = clipped_ratio * atarg_b
                    pol_surr = -torch.min(surr1, surr2).mean()

                # value_loss
                cur_vpred = self.critic(ob_b).squeeze(1)

                # [EXPERIMENT] - value clipping: clipped_value = old_values + (curr_values - old_values).clip(-eps, +eps)
                if self.config['experiment']['value_clipping']:
                    cur_vpred_clipped = old_vpred_b + (cur_vpred - old_vpred_b).clamp(-eps, eps)
                    vloss1 = (cur_vpred - vtarg_b).pow(2)
                    vloss2 = (cur_vpred_clipped - vtarg_b).pow(2)
                    vf_loss = torch.max(vloss1, vloss2).mean()
                else:
                    # original value_loss
                    vf_loss = (cur_vpred - vtarg_b).pow(2).mean()

                # entropy_loss
                pol_entpen = -cur_entropies.mean()

                # total loss
                c1 = self.config['train']['ppo']['coef_vf']
                c2 = self.config['train']['ppo']['coef_entpen']

                # actor - backward
                self.actor_optimizer.zero_grad()
                policy_loss = pol_surr + c2 * pol_entpen
                policy_loss.backward()

                # [EXPERIMENT] - clipping gradient with max_norm=0.5
                if self.config['experiment']['clipping_gradient']:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

                self.actor_optimizer.step()

                # critic - backward
                self.critic_optimizer.zero_grad()
                value_loss = c1 * vf_loss
                value_loss.backward()

                # [EXPERIMENT] - clipping gradient with max_norm=0.5
                if self.config['experiment']['clipping_gradient']:

                    nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.critic_optimizer.step()

                policy_losses.append(pol_surr.item())
                entropy_losses.append(pol_entpen.item())
                value_losses.append(vf_loss.item())

        avg_policy_loss = np.mean(policy_losses)
        avg_value_losses = np.mean(value_losses)
        avg_entropy_losses = np.mean(entropy_losses)
