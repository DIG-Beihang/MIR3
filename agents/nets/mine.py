import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

    
class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func0 = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.LeakyReLU())
        self.T_func1 = nn.Sequential(nn.Linear(y_dim, hidden_size), nn.LeakyReLU())
    
    def forward(self, x_samples, y_samples):
        # samples [sample_size, dim]
        T0 = self.T_func0(x_samples)
        T1 = self.T_func1(y_samples)
        
        N = T0.shape[0]
        
        E = torch.mm(T1, T0.t())
        M = torch.eye(N).to(T0.device)
        
        E0 = math.log(2.) - F.softplus(-E) # positive score
        E1 = F.softplus(-E) + E - math.log(2.) # negative score
        
        return (E0 * M).sum(1) - (E1 * (1 - M)).sum(1) / (N-1)
        

    def learning_loss(self, x_samples, y_samples):
        return -torch.mean(self.forward(x_samples, y_samples))


class MINEOR(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINEOR, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):
        # samples [sample_size, dim]
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))
        
        return T0.mean(1) - torch.log(T1.exp().mean(1))
    
    def learning_loss(self, x_samples, y_samples):
        return -torch.mean(self.forward(x_samples, y_samples))


class MINESP(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINESP, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):
        # samples [sample_size, dim]
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))
        
        E0 = math.log(2.) - F.softplus(-T0)
        E1 = F.softplus(-T1) + T1 - math.log(2.)

        return E0.mean(1) - E1.mean(1)

    def learning_loss(self, x_samples, y_samples):
        return -torch.mean(self.forward(x_samples, y_samples))