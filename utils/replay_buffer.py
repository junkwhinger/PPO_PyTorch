import numpy as np

class PPOMemory:
    def __init__(self, gamma, tau):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []

        self.tdlamret = []
        self.advants = []

        self.gamma = gamma
        self.tau = tau
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, s, a, r, v, lp):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.values.append(v)
        self.logprobs.append(lp)
        self.ptr += 1

    def finish_path(self, v):
        """
        https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py line 64
        """
        path_slice = np.arange(self.path_start_idx, self.ptr)

        rewards_np = np.array(self.rewards)[path_slice]
        values_np = np.array(self.values)[path_slice]
        values_np_added = np.append(values_np, v)

        # GAE
        gae = 0
        advants = []
        for t in reversed(range(len(rewards_np))):
            delta = rewards_np[t] + self.gamma * values_np_added[t+1] - values_np_added[t]
            gae = delta + self.gamma * self.tau * gae
            advants.insert(0, gae)

        self.advants.extend(advants)

        advants_np = np.array(advants)
        tdlamret_np = advants_np + values_np
        self.tdlamret.extend(tdlamret_np.tolist())

        self.path_start_idx = self.ptr

    def reset_storage(self):
        self.ptr, self.path_start_idx = 0, 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.tdlamret = []
        self.advants = []

    def get(self):
        # reset marker
        data = dict(states=self.states, actions=self.actions, logpas=self.logprobs,
                    rewards=self.rewards, values=self.values,
                    tdlamret=self.tdlamret, advants=self.advants)
        self.reset_storage()
        return data

    def __len__(self):
        return len(self.rewards)
