import torch
from lunarlander_dataset import load_demo, get_sa_from_demo, DemoDataset


def get_gail_dataset(exp_name, minimum_score, demo_count, ppo_memory_size, dstep):
    demo = load_demo(exp_name, minimum_score, demo_count)
    expert_sa = get_sa_from_demo(demo)
    batch_size = ppo_memory_size // dstep
    gail_dataset = DemoDataset(expert_sa, batch_size)
    return gail_dataset

def logit_bernoulli_entropy(logits):
    ent = (1. - torch.sigmoid(logits) * logits - torch.log(torch.sigmoid(logits)))
    return ent