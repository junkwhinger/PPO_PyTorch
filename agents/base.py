import os
from abc import *
import torch
import gym


class BaseAgent(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @staticmethod
    def init_env(env_name):
        env = gym.make(env_name)
        return env

    @staticmethod
    def save_weight(model, config_name, weight_name):

        save_path = os.path.join("experiments", config_name, "checkpoints", "ep_{}.pth".format(weight_name))
        weight = model.state_dict()
        torch.save(weight, save_path)

    @staticmethod
    def load_weight(model, config_name, ep=None):
        if not ep:
            filename = "ep_best.pth"
        else:
            filename = "ep_{}.pth".format(ep)
        weight_path = os.path.join("experiments", config_name, "checkpoints", filename)

        if os.path.isfile(weight_path):
            weight = torch.load(weight_path)
            model.load_state_dict(weight)
        else:
            raise FileNotFoundError
