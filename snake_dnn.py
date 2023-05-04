from re import I
import torch
import torch.nn as nn

class DQN(nn.Module):
    # Neural network architecture:
    #
    # input layer: 
    # current_direction
    # food_angle
    # food_distance
    # border_up
    # border_down
    # border_left
    # border_right
    # moves_since_last_food
    #
    # output layer:
    # direction
    def __init__(self, input_dim, hidden_dim, output_dim, board_width, board_height, device):
        super(DQN, self).__init__()
        # Sigmoid sucked, it couldn't turn left
        # An extra layer sucked, it took forever to train or something?
        # RelU worked well
        # LeakyReLU kicked arse, significally better (within 1000 games, things were cooking)
        # Randomized LeakyReLU didn't work as well
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.LeakyReLU()  # ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        # extra layer below, either I'm not letting it train long enough,
        # or it actively prevents progress?
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc6 = nn.ReLU()
        # self.fc7 = nn.Linear(hidden_dim, output_dim)
        self.board_width = board_width
        self.board_height = board_height

        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        # you can another layer or two here, but it wasn't that effective.
        # x = self.fc6(x)
        # x = self.fc7(x)
        return x

    def minmax_scaler(self, e, min, max):
        numerator = (e - min)
        denominator = (max - min)
        return numerator / denominator

    def range_scaler(self, e, lower_bound, upper_bound, min, max):
        numerator = (e - min)
        denominator = (max - min)
        return (upper_bound - lower_bound) * (numerator / denominator) + lower_bound

    def feature_engineer(self, snake_state):
        object_up = self.minmax_scaler(snake_state['object_up'], 1, self.board_height - 2)
        object_down = self.minmax_scaler(snake_state['object_down'], 1, self.board_height - 2)
        object_left = self.minmax_scaler(snake_state['object_left'], 1, self.board_width - 2)
        object_right = self.minmax_scaler(snake_state['object_right'], 1, self.board_width - 2)
        food_angle = self.minmax_scaler(snake_state['food_angle'], 0, 360)
        food_distance = self.minmax_scaler(snake_state['food_distance'],
                                           1,
                                           self.board_width - 2 + self.board_height - 2)
        current_direction = self.minmax_scaler(snake_state['current_direction'], 0, 3)
        moves_since_food = self.minmax_scaler(snake_state['moves_since_last_food'], 1, 50)
        return torch.tensor([current_direction, food_angle, food_distance,
                             object_up, object_down, object_left, object_right],
                            device=self.device, requires_grad=True)

    # feature engineering!
    def feature_y(self, snake_state):
        vec = snake_state['food_direction_distance'].copy()
        for value in sorted(vec):
            index = vec.index(value)
            if (index == 0 and snake_state['object_up'] > 1): return torch.tensor([1., 0., 0., 0.], device=self.device)
            if (index == 1 and snake_state['object_down'] > 1): return torch.tensor([0., 1., 0., 0.], device=self.device)
            if (index == 2 and snake_state['object_left'] > 1): return torch.tensor([0., 0., 1., 0.], device=self.device)
            if (index == 3 and snake_state['object_right'] > 1): return torch.tensor([0., 0., 0., 1.], device=self.device)

        max_distance = [snake_state['object_up'],
                        snake_state['object_down'],
                        snake_state['object_left'],
                        snake_state['object_right']]
        m = max_distance.index(max(max_distance))
        if (m == 0): return torch.tensor([1., 0., 0., 0.], device=self.device)
        if (m == 1): return torch.tensor([0., 1., 0., 0.], device=self.device)
        if (m == 2): return torch.tensor([0., 0., 1., 0.], device=self.device)
        if (m == 3): return torch.tensor([0., 0., 0., 1.], device=self.device)
        return torch.tensor([0., 0., 0., 1.], device=self.device)

    def criterion(self, out, label):
        return torch.sum((label - out)**2)

