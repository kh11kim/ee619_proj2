

# This is recommended skleton code. You can change this file as you want.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(1)


class Net(nn.Module):
    # Q network
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)
        x = self.linear(x)
        return x

#Recommended hyper-parameters. You can change if you want #


class DeepQLearning:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        discount_factor=0.9,
        # e_greedy=0.05,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=20,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        #self.e_greedy = e_greedy
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.construct_network()
        # s(2), a(1), r(1), s'(2), terminal
        self.memory = np.zeros((memory_size, 7))
        self.i = 0
        self.i_max = 0
        self.i_update = 0
    ######################################## TO DO ##############################################
    # Initialzie variables here

    def construct_network(self):
        self.policy_net = Net()
        self.target_net = Net()
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate)
        ######################################## TO DO ##############################################

    def store_transition(self, s, a, r, next_s, terminal):
        experience_tuple = np.array([*s, a, r, *next_s, terminal])
        self.memory[self.i, :] = experience_tuple
        self.i = (self.i + 1) % self.memory_size
        self.i_max = max(self.i_max, self.i)

    def choose_action(self, state, step):
        coin = np.random.random()
        eps_thres = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * step / self.eps_decay)
        if coin > eps_thres:
            with torch.no_grad():
                actions = self.policy_net(state)
                a = np.argmax(actions.numpy())
        else:
            a = np.random.choice([0, 1, 2, 3])
        return a
        ######################################## TO DO ##############################################

    def learn(self):
        sample_indexes = np.random.randint(
            low=0, high=self.i_max, size=self.batch_size)
        state_batch = torch.from_numpy(
            self.memory[sample_indexes, 0:2]).float().to(device)
        action_batch = torch.from_numpy(
            self.memory[sample_indexes, 2]).float().to(device)
        reward_batch = torch.from_numpy(
            self.memory[sample_indexes, 3]).float().to(device)
        next_state_batch = torch.from_numpy(
            self.memory[sample_indexes, 4:6]).float().to(device)
        terminal_batch = torch.from_numpy(
            self.memory[sample_indexes, -1]).float().to(device)

        non_final_mask = ~terminal_batch.type(torch.bool)
        non_final_next_states = next_state_batch[non_final_mask]

        q_value = self.policy_net(state_batch).gather(
            1, action_batch.type(torch.int64).view(-1, 1))
        target_q_value = torch.zeros(self.batch_size)
        target_q_value[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()
        expected_q_value = (
            target_q_value * self.discount_factor) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_value, expected_q_value.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.i_update += 1
        if self.i_update % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        ######################################## TO DO ##############################################
