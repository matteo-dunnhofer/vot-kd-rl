"""
Written by Matteo Dunnhofer - 2020

Class that defines the student model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.ResNets import ResNet18


class StudentModel(torch.nn.Module):

    def __init__(self, cfg):
        super(StudentModel, self).__init__()

        self.model_name = 'StudentModel'

        self.cfg = cfg

        self.lstm_layers = 1
        self.lstm_size = 512

        self.cnn_features = ResNet18(self.cfg)

        self.fc = nn.Linear(8192 * 2, 512)

        self.fc2 = nn.Linear(512, 512)

        self.lstm = nn.LSTM(512, hidden_size=self.lstm_size, num_layers=self.lstm_layers)

        self.actor_policy = nn.Linear(self.lstm_size, 4)

        self.critic = nn.Linear(self.lstm_size, 1)


    def forward(self, x1, x2, state, device):
        x, n_state = self.get_feats(x1, x2, state, device)

        return self.actor_policy(x), self.critic(x), n_state

    def init_state(self, device):
        """
        Returns the initial state of the model
        """
        return (torch.zeros(self.lstm_layers, 1, self.lstm_size).to(device),
                torch.zeros(self.lstm_layers, 1, self.lstm_size).to(device))

    def get_feats(self, x1, x2, state, device):
        """
        Function that executes the model
        """
        x1 = self.cnn_features(x1)
        x2 = self.cnn_features(x2)

        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)

        x = torch.cat([x1, x2], dim=1)

        x = F.relu(self.fc(x))

        x = F.relu(self.fc2(x))

        state = (Variable(state[0].data.to(device)), Variable(state[1].data.to(device)))

        x, n_state = self.lstm(x.unsqueeze(0), state)
        x = x.squeeze(0)

        return x, n_state
