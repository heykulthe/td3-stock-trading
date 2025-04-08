import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size=10000, use_float32=True):

        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        dtype = np.float32 if use_float32 else np.float64

        self.state = np.zeros((self.max_size, state_dim), dtype=dtype)
        self.action = np.zeros((self.max_size, action_dim), dtype=dtype)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=dtype)
        self.reward = np.zeros((self.max_size, 1), dtype=dtype)
        self.not_done = np.zeros((self.max_size, 1), dtype=dtype)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        # Store (1 - done) to multiply with future rewards in TD learning
        self.not_done[self.ptr] = 1.0 - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )