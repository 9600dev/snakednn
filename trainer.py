import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import math
import pprint
import random
import collections
from collections import namedtuple
from itertools import count
from torch.autograd import Variable

# import the snake game simulator
from snake_simulator import SnakeGame, BoardElements, Direction, SnakeState

# import play_game from our inference file
from inference import play_game

# import the pytorch NN for snake
from snake_dnn import DQN


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# use a GPU if it's available, otherwise good 'ol CPU training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 4000
TARGET_UPDATE = 100
BOARD_WIDTH = 30
BOARD_HEIGHT = 10

policy_net = DQN(7, 7, 4, BOARD_WIDTH, BOARD_HEIGHT, device).to(device)
target_net = DQN(7, 7, 4, BOARD_WIDTH, BOARD_HEIGHT, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def calculate_eps_threshold():
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    return eps_threshold


def select_action(game_state: SnakeGame, random_action: bool = False):
    def get_max(t):
        # get the index of the tensor and return that
        r = t.max(0)[1].view(1, 1)
        return r

    def do_random():
        random_index = random.randrange(4)
        random_tensor = torch.zeros(4, device=device)
        random_tensor[random_index] = 1.0
        return random_tensor

    if random_action:
        return get_max(do_random())

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return get_max(policy_net(game_state))  # .max(1)[1].view(1, 1)
    else:
        return get_max(do_random())


episode_durations = collections.deque(maxlen=100)
episode_scores = collections.deque(maxlen=100)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return torch.tensor(0, device=device)
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)  # torch.cat(batch.state)
    action_batch = torch.stack(batch.action)  # torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)  # torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    result = policy_net(state_batch)
    state_action_values = result.gather(1, action_batch.view(-1, 1))

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


def train():
    num_episodes = 500000
    loss = torch.tensor(0, device=device)
    max_score = 0
    max_life = 0

    # this is the training loop
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        game_state = SnakeGame(board_width=BOARD_WIDTH, board_height=BOARD_HEIGHT)

        # kick things off
        state = policy_net.feature_engineer(game_state.get_snake_state())

        for t in count():
            # Select and perform an action
            action = select_action(state)
            returned_state, reward, done = game_state.next_move_action(action)

            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = policy_net.feature_engineer(game_state.get_snake_state())
            else:
                next_state = None

            # Store the transition in memory
            memory.push(torch.tensor(state, requires_grad=True, device=device), action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = optimize_model()
            max_score = max(max_score, game_state.score)
            max_life = max(max_life, game_state.life_counter)
            # previous_reward = tensor_reward

            if done:
                episode_durations.append(t + 1)
                episode_scores.append(game_state.score)
                break

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 100 == 0:
            print('games played: {} max_score: {} max_life: {} ' \
                  'average_duration: {} average_score: {} eps_threshold: {} loss: {}'
                  .format(str(i_episode),
                          max_score,
                          max_life,
                          round(np.average(episode_durations), 2),
                          round(np.average(episode_scores), 2),
                          round(calculate_eps_threshold(), 3),
                          round(loss.item(), 25)))

        if i_episode % 500 == 0 and i_episode != 0:
            # show the user a game, displaying the NN's almighty learning power!
            play_game(policy_net, device, BOARD_WIDTH, BOARD_HEIGHT)

            # checkpoint the current training state
            torch.save(policy_net, 'training_state.pt')


if __name__ == '__main__': 
    train()
