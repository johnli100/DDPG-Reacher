import numpy as np
import torch
from torch import optim,nn
from collections import deque,namedtuple
import random

MEMORY_SIZE = 10000
GAMMA = 0.995
TAU = 0.005
LR = 0.0005


Transition = namedtuple('Transition',('state','action','reward','state_next','done'))

class TD3Agent():

    def __init__(self, state_dim=37, action_dim=4):
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_target = Network_Actor(self.state_dim,self.action_dim).to(self.device)
        for param in self.actor_target.parameters():
            param.requires_grad = False
        self.critic_target = Network_Critic(self.state_dim,self.action_dim).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        self.actor = Network_Actor(self.state_dim,self.action_dim).to(self.device)
        self.critic = Network_Critic(self.state_dim,self.action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)

        self.memory = deque(maxlen=MEMORY_SIZE)

    def cache(self, state, action, reward, state_next, done):
        state = torch.tensor(data=state, dtype=torch.float, device=self.device).squeeze(0)
        action = torch.tensor(action, dtype=torch.float, device=self.device).squeeze(0)
        reward = torch.tensor(reward, dtype=torch.float,device=self.device).unsqueeze(0)
        state_next = torch.tensor(state_next,dtype=torch.float,device=self.device).squeeze(0)
        done = torch.tensor(done, dtype=torch.float, device=self.device).unsqueeze(0)
        self.memory.append(Transition(state, action, reward, state_next, done))

    def recall(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, states_next, dones = map(torch.stack, zip(*transitions))
        return (states, actions, rewards, states_next, dones)

    def act(self, state, policy_noise):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = self.actor(state).data.numpy()
        # random noise and clip
        action += np.clip(np.random.randn() * policy_noise, -0.5, 0.5)
        return action.clip(-1., 1.)

    def learn(self, transitions):
        states, actions, rewards, states_next, dones = transitions
        # learn Value function (critic)
        Qs_1, Qs_2 = self.critic(states, actions)
        actions_next = self.actor_target(states_next)
        Qs_1_target, Qs_2_target = self.critic_target(states_next, actions_next)
        Qs_target = torch.min(Qs_1_target.data, Qs_2_target.data) * GAMMA * (1-dones) + rewards
        loss_critic = nn.functional.mse_loss(Qs_1, Qs_target) + nn.functional.mse_loss(Qs_2, Qs_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # learn (optimized continuous) policy function (actor)
        loss_actor = -self.critic(states, self.actor(states))[0].mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

    def sync_target(self, soft_update=True):
        if soft_update:
            for target_p, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_p.data.copy_(TAU * p.data + (1 - TAU) * target_p.data)
            for target_p, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_p.data.copy_(TAU * p.data + (1 - TAU) * target_p.data)
        else:
            self.actor_target.load(self.actor.state_dict())
            self.critic_target.load(self.critic.state_dict())


class Network_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300]):
        super().__init__()
        hidden_dims = [state_dim] + hidden_dims
        self.layers = nn.ModuleList([nn.Linear(hidden_dims[i],hidden_dims[i+1]) \
                                      for i in range(len(hidden_dims)-1)])
        self.layer_output = nn.Linear(hidden_dims[-1], action_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return self.tanh(self.layer_output(x))

class Network_Critic(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_dims = [400, 300]):
        super().__init__()
        hidden_dims = [state_dim+action_dim] + hidden_dims
        self.layers_1 = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i + 1]) \
                                      for i in range(len(hidden_dims)-1)])
        self.layer_output_1 = nn.Linear(hidden_dims[-1], 1)

        self.layers_2 = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i + 1]) \
                                       for i in range(len(hidden_dims) - 1)])
        self.layer_output_2 = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state, action):
        x1 = x2 = torch.cat([state, action], dim=1)

        for layer in self.layers_1:
            x1 = layer(x1)
            x1 = nn.functional.relu(x1)

        for layer in self.layers_2:
            x2 = layer(x2)
            x2 = nn.functional.relu(x2)

        return self.layer_output_1(x1), self.layer_output_2(x2)
