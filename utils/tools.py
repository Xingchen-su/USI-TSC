import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

def get_action(time: int, flag: int, action_space: dict, action_dict: dict, agent_name: object):
    '''fixed-time 用，每 flag 秒切换一次所有相位'''
    if time % flag == 0 and time > 0:
        for agt in agent_name:
            if (action_dict[agt] + 1) in action_space[agt]:
                action_dict[agt] += 1
            else:
                action_dict[agt] = 0
            action_dict[agt] = action_space[agt][action_dict[agt]]
    else:
        pass
    return action_dict


def smooth(data: pd.Series, alpha: float = 0.05):
    if isinstance(data, list):
        data = pd.Series(data)
    smooth_data = pd.Series(data).ewm(alpha=alpha).mean()
    return smooth_data


def MARLWrap(type, alg, alg_args, policy, value, state_dim, hidden_dim, action_dim, **kwargs):
    '''
    - type: should be one of ['CTDE', 'I'] \n
    `CTDE`: CTDE agents like MAPPO \n
    `I`: independent agents like IPPO \n

    - alg: algorithm, should be a agent algorithm
    - alg_args: parameters of alg
    '''
    assert type in ['CTDE', 'I'], "type error!"

    if type == 'CTDE':
        assert isinstance(state_dim, list), "state dim should be a list"
        agents = alg(alg_args['agent_name'], policy, value, state_dim, hidden_dim, action_dim,
                     alg_args['actor_lr'], alg_args['critic_lr'], alg_args['gamma'], alg_args['lmbda'],
                     alg_args['epochs'], alg_args['eps'], alg_args['device'])
        return agents

    elif type == 'I':
        assert isinstance(state_dim, int), "state dim should be a integer"
        latent_dim = kwargs.get('latent_dim', 0)
        policy_net = policy(state_dim, hidden_dim, action_dim)
        critic_net = value(state_dim, hidden_dim, latent_dim)
        agents = {}
        for name in alg_args['agent_name']:
            agents[name] = alg(policy_net, critic_net, alg_args['actor_lr'],
                              alg_args['critic_lr'], alg_args['gamma'],
                              alg_args['lmbda'], alg_args['epochs'],
                              alg_args['eps'], alg_args['device'])
        return agents


class ReplayBuffer:
    """CEA."""

    def __init__(
        self,
        agent_num: int,
        states_dim: int,
        size: int,
        batch_size: int = 128,
    ):
        self.states_buf = torch.zeros([size, agent_num, states_dim], dtype=torch.float32)
        self.next_states_buf = torch.zeros([size, agent_num, states_dim], dtype=torch.float32)
        self.actions_buf = torch.zeros([size, agent_num], dtype=torch.float32)
        self.rewards_buf = torch.zeros([size, agent_num], dtype=torch.float32)
        self.dones_buf = torch.zeros([size, agent_num], dtype=torch.float32)
        self.truncated_buf = torch.zeros([size, agent_num], dtype=torch.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.capacity = size

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        truncated: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]:
        transition = (state, action, reward, next_state, done, truncated)
        self.states_buf[self.ptr] = state
        self.next_states_buf[self.ptr] = next_state
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.dones_buf[self.ptr] = done
        self.truncated_buf[self.ptr] = truncated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return transition

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        idxs = torch.randperm(self.size)[:self.batch_size]
        return dict(
            states=self.states_buf[idxs],
            next_states=self.next_states_buf[idxs],
            actions=self.actions_buf[idxs],
            rewards=self.rewards_buf[idxs],
            dones=self.dones_buf[idxs],
            truncated=self.truncated_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size
