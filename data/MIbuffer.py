import torch
import numpy as np

 
class MIBuffer:
    """buffer for superiority and inferiority trajectories"""
    def __init__(self, pos_buffer_size, neg_buffer_size, n_agent, state_shape, action_shape):
        # state: [n_agent, n_state]
        # action: [n_agent, n_action]
        self.pos_size = pos_buffer_size
        self.neg_size = neg_buffer_size
        self.type = torch.float32
        
        # TODO: In runner, when to reset the PMIC buffer?
        self.pos_state = torch.zeros((self.pos_size, n_agent, state_shape))
        self.pos_action = torch.zeros((self.pos_size, n_agent, action_shape))
        self.pos_reward = []
        self.pos_pointer = 0

        self.neg_state = torch.zeros((self.neg_size, n_agent, state_shape))
        self.neg_action = torch.zeros((self.neg_size, n_agent, action_shape))
        self.neg_reward = []
        self.neg_pointer = 0

    def sort_pos(self):
        if self.pos_reward == []:
            return -1000
        index = torch.tensor(self.pos_reward).argsort().numpy().tolist()
        self.pos_reward = torch.tensor(self.pos_reward)[index].numpy().tolist()
        self.pos_state[:len(self.pos_reward)] = self.pos_state[index]
        self.pos_action[:len(self.pos_reward)] = self.pos_action[index]
        self.pos_pointer = 0
        return self.pos_reward[0]  # return the smallest reward in array
    
    def count_pos(self):
        return len(self.pos_reward)

    def add_pos(self, state, action, reward):
        state = state.detach().cpu().clone().type(self.type)
        action = action.detach().cpu().clone().type(self.type)
        reward = reward.item()
        self.pos_state[self.pos_pointer] = state
        self.pos_action[self.pos_pointer] = action
        if len(self.pos_reward) == self.pos_size:
            self.pos_reward[self.pos_pointer] = reward
        else:
            self.pos_reward.append(reward)
        self.pos_pointer = (self.pos_pointer + 1) % self.pos_size

    def add_neg(self, state, action, reward):
        state = state.detach().cpu().clone().type(self.type)
        action = action.detach().cpu().clone().type(self.type)
        reward = reward.item()
        self.neg_state[self.neg_pointer] = state
        self.neg_action[self.neg_pointer] = action
        if len(self.neg_reward) == self.neg_size:
            self.neg_reward[self.neg_pointer] = reward
        else:
            self.neg_reward.append(reward)
        self.neg_pointer = (self.neg_pointer + 1) % self.neg_size

    def sample_pos(self, batch_size):
        index = torch.randint(0, len(self.pos_reward) - 1, size=(batch_size,))
        return self.pos_state[index], self.pos_action[index]

    def sample_neg(self, batch_size):
        index = torch.randint(0, len(self.neg_reward) - 1, size=(batch_size,))
        return self.neg_state[index], self.neg_action[index]