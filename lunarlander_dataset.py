import os
import numpy as np
import torch


class DemoDataset:
    def __init__(self, demo, batch_size):
        self.demo = demo
        self.batch_size = batch_size
        self.total_demo_size = self.demo[0].size(0)
        self.nb_batches = self.total_demo_size // self.batch_size
        self.demo_indices = np.arange(self.total_demo_size)
        np.random.shuffle(self.demo_indices)
        self.init_pointer = 0

    def get_next_batch(self):
        # if not enough demo left, reset it
        if self.init_pointer + self.batch_size > self.total_demo_size:
            self.init_pointer = 0
            np.random.shuffle(self.demo_indices)

        # get demo states and actions
        demo_states = self.demo[0][self.init_pointer: self.init_pointer + self.batch_size]
        demo_actions = self.demo[1][self.init_pointer: self.init_pointer + self.batch_size]

        # increment init_pointer
        self.init_pointer += self.batch_size

        return demo_states, demo_actions


def load_demo(exp_name, minimum_score, demo_count):
    demo_path = os.path.join("experiments", exp_name, "demonstration", "demo.pth")
    demo = torch.load(demo_path)

    valid_trajectories = []
    for trajectory, score in demo:
        if score >= minimum_score:
            valid_trajectories.append(trajectory)

        if len(valid_trajectories) >= demo_count:
            break

    return valid_trajectories

def get_sa_from_demo(demo):

    tmp = []
    for trajectory in demo:
        states = [torch.tensor(step[0]).float() for step in trajectory]
        actions = [torch.tensor(step[1]).long() for step in trajectory]

        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        tmp.append((states_tensor, actions_tensor))

    states_tmp = torch.cat([traj[0] for traj in tmp])
    actions_tmp = torch.cat([traj[1] for traj in tmp])

    return states_tmp, actions_tmp
