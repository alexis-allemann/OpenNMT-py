
import numpy as np
import torch.distributed
from onmt.schedulers import register_scheduler
from onmt.utils.logging import logger
from onmt.utils.distributed_new import all_gather_list
from .scheduler import Scheduler

import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Implementation of Deep Q-Network (DQN) algorithm inspired from PyTorch's tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        return self.layer3(x)

@register_scheduler(scheduler="dqn")
class DQNScheduler(Scheduler):
    """DQN scheduling class."""

    def __init__(self, nb_actions, nb_states, opts, device_id) -> None:
        super().__init__(nb_actions, nb_states, opts, device_id)
        self.lastReward = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action = torch.zeros((1,1), dtype=torch.int, device=self.device)
        default_state = np.zeros(nb_states)
        self.last_state = torch.tensor(default_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.memory = ReplayMemory(self.MAX_REPLAY_CAPACITY)
        self.exploration = True
        self.Q = None
        self.policy_net = DQN(self.nb_states, self.nb_actions, self.HIDDEN_SIZE).to(self.device)
        self.target_net = DQN(self.nb_states, self.nb_actions, self.HIDDEN_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.LR)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to this Curriculum."""
        super().add_options(parser)
        group = parser.add_argument_group("DQN")
        group.add_argument(
            "-dqn_batch_size",
            "--dqn_batch_size",
            type=int,
            default=128,
            help="Batch size for the DQN algorithm.",
        )
        group.add_argument(
            "-dqn_tau",
            "--dqn_tau",
            type=float,
            default=0.005,
            help="Tau for the soft update of the target network.",
        )
        group.add_argument(
            "-dqn_min_replay_capacity",
            "--dqn_min_replay_capacity",
            type=int,
            default=3000,
            help="Minimum replay capacity for the DQN algorithm.",
        )
        group.add_argument(
            "-dqn_max_replay_capacity",
            "--dqn_max_replay_capacity",
            type=int,
            default=10000,
            help="Maximum replay capacity for the DQN algorithm.",
        )
        group.add_argument(
            "-dqn_eps_start",
            "--dqn_eps_start",
            type=float,
            default=1,
            help="Start value for epsilon.",
        )
        group.add_argument(
            "-dqn_eps_end",
            "--dqn_eps_end",
            type=float,
            default=0.01,
            help="End value for epsilon.",
        )
        group.add_argument(
            "-dqn_eps_decay",
            "--dqn_eps_decay",
            type=int,
            default=100000,
            help="Decay for epsilon.",
        )
        group.add_argument(
            "-dqn_gamma",
            "--dqn_gamma",
            type=float,
            default=0.99,
            help="Gamma for the DQN algorithm.",
        )
        group.add_argument(
            "-dqn_lr",
            "--dqn_lr",
            type=float,
            default=0.00025,
            help="Learning rate for the DQN algorithm.",
        )
        group.add_argument(
            "-dqn_hidden_size",
            "--dqn_hidden_size",
            type=int,
            default=512,
            help="Hidden size for the DQN algorithm.",
        )

    @classmethod
    def _validate_options(cls, opts):
        super()._validate_options(opts)

    def _parse_opts(self):
        super()._parse_opts()
        self.BATCH_SIZE = self.opts.dqn_batch_size
        self.TAU = self.opts.dqn_tau
        self.MIN_REPLAY_CAPACITY = self.opts.dqn_min_replay_capacity
        self.MAX_REPLAY_CAPACITY = self.opts.dqn_max_replay_capacity
        self.EPS_START = self.opts.dqn_eps_start
        self.EPS_END = self.opts.dqn_eps_end
        self.EPS_DECAY = self.opts.dqn_eps_decay
        self.GAMMA = self.opts.dqn_gamma
        self.LR = self.opts.dqn_lr
        self.HIDDEN_SIZE = self.opts.dqn_hidden_size

    def select_action(self, step, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * step / self.EPS_DECAY)
        if sample > eps_threshold:
            self.exploration = False
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                predictions = self.policy_net(state)
                self.Q = predictions.squeeze(0)
                return predictions.max(1).indices.view(1, 1)
        else:
            self.exploration = True
            self.Q = None
            return torch.tensor([[random.randint(0, self.nb_actions-1)]], device=self.device, dtype=torch.long)
        
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE or len(self.memory) < self.MIN_REPLAY_CAPACITY:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def next_task(self, step, reward, state):
        super().next_task(step, reward, state)

        if self.device_id == 0:
            self.delta_reward = self.lastReward - reward
            self.lastReward = reward
            delta_reward = torch.tensor([self.delta_reward], device=self.device)

            if step >= self.warmup_steps:
                next_state = state.clone().detach().unsqueeze(0)

                # Store the transition in memory
                self.memory.push(self.last_state, self.action, next_state, delta_reward)

                # Move to the next state
                self.last_state = next_state

                if len(self.memory) >= self.MIN_REPLAY_CAPACITY:
                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                    self.target_net.load_state_dict(target_net_state_dict)
                    self.action = self.select_action(step, next_state)
                else:
                    self.action = torch.tensor([[np.random.choice(self.nb_actions)]], device=self.device, dtype=torch.long)
            else:
                available_actions = list(range(self.nb_actions))
                if self.hrl_warmup:
                    available_actions = self.hrl_wamup_tasks
                self.action = torch.tensor([[np.random.choice(available_actions)]], device=self.device, dtype=torch.long)
            self._log(step)
        
        actions = all_gather_list(self.action) # Gather the actions from all GPUs
        self.action = actions[0].view(1,1) # Select the action from the first GPU
        return self.action.item()
    
    def _log(self, step):
        qvalues = "None"
        if self.Q is not None:
            qvalues = "["
            for i in range(self.nb_actions):
                qvalues += f"{self.Q[i]}"
                if i < self.nb_actions - 1:
                    qvalues += ", "
            qvalues += "]"
        logger.info(f"Step:{step+1};GPU:{self.device_id};Q-values:{qvalues};Action:{self.action.item()};Delta-reward:{self.delta_reward};Exploration:{self.exploration}")

    def save_scheduler(self, path):
        torch.save(self.policy_net.state_dict(), path)
        logger.info(f"Model saved to {path}")