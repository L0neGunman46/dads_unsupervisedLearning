from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_filt, skill, action, reward, next_state_filt, done,
             state_full=None, next_state_full=None):
        # fall back if full versions not provided
        if state_full is None:
            state_full = state_filt
        if next_state_full is None:
            next_state_full = next_state_filt
        self.buffer.append(
            (state_filt, skill, action, reward, next_state_filt, done,
             state_full, next_state_full)
        )

    def sample(self, batch_size):
        items = random.sample(self.buffer, batch_size)
        (state_f, skill, action, reward, next_state_f, done,
         state_full, next_state_full) = zip(*items)
        return (
            np.array(state_f),
            np.array(skill),
            np.array(action),
            np.array(reward).reshape(-1, 1),
            np.array(next_state_f),
            np.array(done).reshape(-1, 1),
            np.array(state_full),
            np.array(next_state_full),
        )

    def __len__(self):
        return len(self.buffer)