import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils.custom_activation_fn import swish

def init_normal_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def init_orthogonal_weights(m):
    if isinstance(m, nn.Linear):
        orthogonal_init(m.weight)
        nn.init.constant_(m.bias, 0.1)

def orthogonal_init(tensor, gain=1):
    '''
    https://github.com/implementation-matters/code-for-paper/blob/094994f2bfd154d565c34f5d24a7ade00e0c5bdb/src/policy_gradients/torch_utils.py#L494
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

class Actor(nn.Module):
    def __init__(self, device, input_dim, output_dim, hidden_dims, hidden_activation_fn, weight_init_scheme="normal"):
        super(Actor, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx+1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        if hidden_activation_fn == "tanh":
            self.hfn = torch.tanh
        elif hidden_activation_fn == 'relu':
            self.hfn = torch.relu
        elif hidden_activation_fn == 'swish':
            self.hfn = swish
        else:
            raise NotImplementedError

        if weight_init_scheme == "normal":
            self.apply(init_normal_weights)
        elif weight_init_scheme == "orthogonal":
            self.apply(init_orthogonal_weights)
        else:
            raise ValueError

        self.device = device

    def select_action(self, states):
        # sample action
        probs = self.forward(states)
        dist = Categorical(probs=probs)
        actions = dist.sample()

        # log prob of that action
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def select_greedy_action(self, states):
        # select action with the highest prob
        probs = self.forward(states)
        _, actions = probs.max(1)
        return actions

    def get_predictions(self, states, old_actions):
        # get log_probs of old actions and current entropy of the distribution
        state, old_actions = self._format(states), self._format(old_actions)
        probs = self.forward(states)
        dist = Categorical(probs=probs)

        log_probs = dist.log_prob(old_actions)
        entropies = dist.entropy()
        return log_probs, entropies

    def forward(self, state):
        """return action probabilities given state"""
        state = self._format(state)

        x = self.input_layer(state)
        x = self.hfn(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.hfn(x)

        x = self.output_layer(x)
        x = torch.softmax(x, dim=1)
        return x

    def _format(self, state):
        """convert numpy state to tensor and add batch dim"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.unsqueeze(0) # add bsz dim if state is in numpy array
        return state


class Critic(nn.Module):
    def __init__(self, device, input_dim, hidden_dims, hidden_activation_fn, weight_init_scheme="normal"):
        super(Critic, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for idx in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        if hidden_activation_fn == "tanh":
            self.hfn = torch.tanh
        elif hidden_activation_fn == 'relu':
            self.hfn = torch.relu
        elif hidden_activation_fn == 'swish':
            self.hfn = swish
        else:
            raise NotImplementedError

        if weight_init_scheme == "normal":
            self.apply(init_normal_weights)
        elif weight_init_scheme == "orthogonal":
            self.apply(init_orthogonal_weights)
        else:
            raise ValueError

        self.device = device

    def forward(self, state):
        """return estimated value given state"""
        state = self._format(state)

        x = self.input_layer(state)
        x = self.hfn(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.hfn(x)

        x = self.output_layer(x)
        return x

    def _format(self, state):
        """convert numpy state to tensor and add batch dim"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.unsqueeze(0)  # add bsz dim if state is in numpy array
        return state
