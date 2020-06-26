from operator import itemgetter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lunarlander_dataset import load_demo, get_sa_from_demo, DemoDataset
from utils.stuff import ObservationScaler

def get_bc_dataset(exp_name, minimum_score, batch_size, demo_count, val_size=0.1):
    demo = load_demo(exp_name, minimum_score, demo_count)
    traj_indices = np.arange(len(demo))
    np.random.shuffle(traj_indices)
    train_num = int(len(demo) * (1. - val_size))
    train_indices = traj_indices[:train_num]
    val_indices = traj_indices[train_num:]

    train_demos = itemgetter(*train_indices)(demo)
    val_demos = itemgetter(*val_indices)(demo)

    train_expert_sa = get_sa_from_demo(train_demos)
    val_expert_sa = get_sa_from_demo(val_demos)
    val_total_size = val_expert_sa[0].size(0)

    train_dataset = DemoDataset(train_expert_sa, batch_size)
    val_dataset = DemoDataset(val_expert_sa, val_total_size)
    return train_dataset, val_dataset

def pretrain(policy, policy_lr, epochs, train_set, valid_set, use_obs_scaler=False, writer=None):
    policy_optim = optim.Adam(policy.parameters(), lr=policy_lr)
    if use_obs_scaler:
        observation_scaler = ObservationScaler()

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        # train
        train_n = 0
        train_correct = 0
        policy.train()
        for _ in range(train_set.nb_batches):
            states, actions = train_set.get_next_batch()
            if use_obs_scaler:
                states = torch.stack([observation_scaler(state) for state in states])

            logits = policy.forward(states)
            probs = torch.softmax(logits, dim=1)
            _, prediction = probs.max(1)
            train_correct += prediction.eq(actions).sum().item()
            train_n += states.size(0)

            loss = loss_fn(logits, actions)
            policy_optim.zero_grad()
            loss.backward()
            policy_optim.step()

        # eval
        valid_n = 0
        valid_correct = 0
        policy.eval()
        for _ in range(valid_set.nb_batches):
            states, actions = valid_set.get_next_batch()
            if use_obs_scaler:
                observation_scaler = ObservationScaler()
                states = torch.stack([observation_scaler(state, update=False) for state in states])

            with torch.no_grad():
                logits = policy.forward(states)
            probs = torch.softmax(logits, dim=1)
            _, prediction = probs.max(1)
            valid_correct += prediction.eq(actions).sum().item()
            valid_n += states.size(0)

        train_acc = train_correct / train_n
        valid_acc = valid_correct / valid_n

        print("[BC info] epoch: {} \t train accuracy: {:.3f} \t valid accuracy: {:.3f}".format(epoch, train_acc, valid_acc))
        if writer:
            writer.add_scalars("info/bc_accuracy", {'train_accuracy': train_acc,
                                                    'valid_accuracy': valid_acc}, epoch)

    policy.train()
    return policy


# BC test
# from models import Actor
# import torch.optim as optim
#
# actor = Actor(device="cpu",
#               input_dim=8, output_dim=4,
#               hidden_dims=[32, 32],
#               hidden_activation_fn='relu',
#               weight_init_scheme='normal')
#
# tr_set, va_set = get_bc_dataset("PPO_M", 230, 64, 200)
# pretrain(actor, 0.002, 10, tr_set, va_set, False)

# all = load_demo_dataset("gail")
# train, val = load_demo_dataset("bc")

# load trajs
#

#
# def pretrain(policy, demo, writer=None, epochs=10):
#     bc_optim = optim.Adam(policy.parameters(), lr=0.002)
#
#     print(len(demo.demo))


# actor = Actor(device="cpu",
#               input_dim=8, output_dim=4,
#               hidden_dims=[32, 32],
#               hidden_activation_fn='relu',
#               weight_init_scheme='normal')

# pretrain(actor, expert_dataset)