import os
import torch

def load_demo(exp_name, minimum_score, demo_count):
    demo_path = os.path.join("experiments", exp_name, "demonstration", "demo.pth")
    demo = torch.load(demo_path)

    valid_trajectories = []
    for trajectory, score in demo:
        if score >= minimum_score:
            states = [torch.tensor(step[0]).float() for step in trajectory]
            actions = [torch.tensor(step[1]).long() for step in trajectory]

            states_tensor = torch.stack(states)
            actions_tensor = torch.stack(actions)
            valid_trajectories.append((states_tensor, actions_tensor))

        if len(valid_trajectories) >= demo_count:
            break

    states_tmp = torch.cat([traj[0] for traj in valid_trajectories])
    actions_tmp = torch.cat([traj[1] for traj in valid_trajectories])

    return states_tmp, actions_tmp

def logit_bernoulli_entropy(logits):
    ent = (1. - torch.sigmoid(logits) * logits - torch.log(torch.sigmoid(logits)))
    return ent