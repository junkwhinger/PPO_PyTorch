import os
import shutil
import numpy as np
import torch

def prepare_dir(exp_name, overwrite=False):

    exp_dir = os.path.join("experiments", exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    tb_dir = os.path.join(exp_dir, "runs")

    if overwrite:
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir)
    os.makedirs(checkpoint_dir)
    os.makedirs(tb_dir)


class RunningStat:
    """
    https://www.johndcook.com/blog/standard_deviation/
    """
    def __init__(self):
        self.__m_n = 0
        self.__m_oldM = 0
        self.__m_newM = 0
        self.__m_oldS = 0
        self.__m_newS = 0

    def reset(self):
        self.__m_n = 0

    def push(self, x):
        self.__m_n += 1

        if self.__m_n == 1:
            self.__m_oldM = x
            self.__m_newM = x
            self.__m_oldS = 0.
        else:
            self.__m_newM = self.__m_oldM + (x - self.__m_oldM) / self.__m_n
            self.__m_newS = self.__m_oldS + (x - self.__m_oldM) * (x - self.__m_newM)

            # set up for next iteration
            self.__m_oldM = self.__m_newM
            self.__m_oldS = self.__m_newS

    def get_num_data_values(self):
        return self.__m_n

    def mean(self):
        if self.__m_n > 0:
            return self.__m_newM
        else:
            return 0

    def variance(self):
        if self.__m_n > 1:
            return self.__m_newS / (self.__m_n - 1)
        else:
            return 0.

    def std(self):
        return np.sqrt(self.variance())

    def load_variables(self, saved_dict):
        self.__m_n = saved_dict['n']
        self.__m_oldM = saved_dict['mu']
        self.__m_newM = saved_dict['mu']
        self.__m_oldS = saved_dict['var']
        self.__m_newS = saved_dict['var']

    def save_variables(self):
        return {"n": self.__m_n,
                "mu": self.__m_newM,
                "var": self.__m_newS}


class RewardScaler:
    def __init__(self, gamma=1.):
        self.rs = RunningStat()
        self.gamma = gamma

    def __call__(self, r_t, update=True):
        if update:
            R_t = self.gamma * self.rs.std() + r_t
            self.rs.push(R_t)
        return r_t / (self.rs.std() + 1e-5)

    def save(self, exp_name):
        save_path = os.path.join("experiments", exp_name, "checkpoints", "reward_scaler.pth")
        print(save_path)
        torch.save(self.rs.save_variables(), save_path)

    def load(self, exp_name):
        load_path = os.path.join("experiments", exp_name, "checkpoints", "reward_scaler.pth")
        self.rs.load_variables(torch.load(load_path))


class ObservationScaler:
    def __init__(self):
        self.rs = RunningStat()

    def __call__(self, ob_t, update=True):
        if update:
            self.rs.push(ob_t)
        return (ob_t - self.rs.mean()) / (self.rs.std() + 1e-5)

    def save(self, exp_name):
        save_path = os.path.join("experiments", exp_name, "checkpoints", "observation_scaler.pth")
        torch.save(self.rs.save_variables(), save_path)

    def load(self, exp_name):
        load_path = os.path.join("experiments", exp_name, "checkpoints", "observation_scaler.pth")
        self.rs.load_variables(torch.load(load_path))


# ObservationScaler test
# ob_s = ObservationScaler()
# for i in range(2):
#     ob = np.array([1, 8]) + np.random.normal(0, 0.5)
#     print("--", ob)
#     a = ob_s(ob)
#
# print(ob_s.rs.mean(), ob_s.rs.std())
# print(ob_s(np.array([[1, 8], [1,8]]), update=False))
# ob_s.save("PPO_M")
#
# ob_s2 = ObservationScaler()
# ob_s2.load("PPO_M")
# print(ob_s2(np.array([100, 800]), update=False))



# RewardScaler test
# rrs = RewardScaler(gamma=0.99)
# for i in range(100):
#     rrs(i, update=True)
# print(rrs(1000, update=False))
# rrs.save("PPO_M")
#
# rrs2 = RewardScaler(gamma=0.99)
# rrs2.load("PPO_M")
# print(rrs2(1000, update=False))


# RunningStat test
# manual_rs = []
# rs = RunningStat()
# for i in np.random.normal(0, 1, 100):
#     manual_rs.append(i)
#     rs.push(i)
#     print(i, np.mean(manual_rs), rs.mean(), np.std(manual_rs), rs.std())
