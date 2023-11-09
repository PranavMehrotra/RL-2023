import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size=64):
        super(Policy, self).__init__()
        self.input_size = state_dim
        self.output_size = num_actions
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

        self.log_std = nn.Parameter(torch.zeros(1, num_actions))
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, log_std, std
    

class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(Value, self).__init__()
        self.input_size = state_dim
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)
    