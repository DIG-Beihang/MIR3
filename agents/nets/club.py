import torch
import torch.nn as nn

class CLUBCategorical(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        '''
        x_dim : the dimension of vector embeddings
        y_dim : the number of categorical labels
        '''
        super(CLUBCategorical, self).__init__()
        
        self.varnet = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim)
        )
        
    def forward(self, x_samples, y_samples):
        # x_samples [sample_size, x_dim]
        # y_samples [sample_size, y_dim]
        
        logits = self.varnet(x_samples)
        
        # log of conditional probability of positive sample pairs
        sample_size, y_dim = logits.shape
        
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)
        y_samples_extend = y_samples.unsqueeze(0).repeat(sample_size, 1, 1)

        import gc

        gc.collect()
        torch.cuda.empty_cache()
        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, y_dim),
            y_samples_extend.reshape(-1, y_dim).argmax(dim=-1),
            reduction='none'
        )
        
        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat)
        negative = log_mat.mean(1)
        return positive - negative

    def loglikeli(self, x_samples, y_samples):
        logits = self.varnet(x_samples)
        return - nn.functional.cross_entropy(logits, y_samples.argmax(dim=-1), reduction="none")
    
    def learning_loss(self, x_samples, y_samples):
        return - torch.mean(self.loglikeli(x_samples, y_samples))
    

class CLUBContinuous(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBContinuous, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2), 
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2), 
                                      nn.ReLU(), 
                                      nn.Linear(hidden_size//2, y_dim), 
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = -(mu - y_samples)**2 /2./logvar.exp()
        
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)

        # log of conditional probability of negative sample pairs
        negative = -((y_samples_1 - prediction_1)**2).mean(1)/2./logvar.exp()

        return positive.sum(1) - negative.sum(1)

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(1)
    
    def learning_loss(self, x_samples, y_samples):
        return - torch.mean(self.loglikeli(x_samples, y_samples))
