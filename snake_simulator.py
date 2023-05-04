import curses
import numpy as np
import sys
import math
import torch
import random
from enum import IntEnum
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint


class BoardElements(IntEnum):
    BORDER = 1
    SNAKE = 2
    FOOD = 3


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SnakeState(IntEnum):
    ALIVE = 0
    DEAD = 1


class SnakeGame:
    """SnakeGame holds all the game state for the snake game
    """
    def __init__(self, board_width=60, board_height=20,
                 curses_visualization=False):
        # state
        if board_width < 15 or board_height < 10:
            raise ValueError('board_width must be >= 15 and board_height must be >= 12')
        self.score = 0
        self.snake = [[3, 6], [3, 5], [3, 4]]
        self.food = []
        self.board_width = board_width
        self.board_height = board_height

        self.previous_direction: Direction = Direction.RIGHT
        self.state = SnakeState.ALIVE
        self.board_matrix = np.matrix(np.zeros(shape=(self.board_height,
                                                      self.board_width)))
        self.life_counter = 0
        self.moves_since_last_food = 0

        # visualization
        self.win = None

        # randomize generation of food, make sure it's not in the snake
        while self.food == []:
            # Calculating next food's coords
            self.food = [randint(1, self.board_height - 2), randint(1, self.board_width - 2)]
            if self.food in self.snake: self.food = []

        # draw border in the matrix
        for i in range(self.board_width):
            self.board_matrix[0, i] = BoardElements.BORDER.value
            self.board_matrix[self.board_height - 1, i] = BoardElements.BORDER.value

        for i in range(self.board_height):
            self.board_matrix[i, 0] = BoardElements.BORDER.value
            self.board_matrix[i, self.board_width - 1] = BoardElements.BORDER.value

        # draw the food and snake
        self.board_matrix[self.food[0], self.food[1]] = BoardElements.FOOD.value
        for s in self.snake:
            self.board_matrix[s[0], s[1]] = BoardElements.SNAKE

        if curses_visualization:
            self.init_curses_board()

    def get_board_matrix(self):
        return self.board_matrix

    def move_snake(self, snake, direction: Direction):
        return [snake[0][0] + (direction == Direction.DOWN and 1) + (direction == Direction.UP and -1),
                snake[0][1] + (direction == Direction.LEFT and -1) + (direction == Direction.RIGHT and 1)]

    def get_distance_to_object(self, snake, direction: Direction):
        x = snake[0][0]
        y = snake[0][1]

        counter = 0
        hit_object = False
        while (not hit_object and x > 0 and y > 0):
            counter = counter + 1
            x = x + (direction == Direction.DOWN and 1) + (direction == Direction.UP and -1)
            y = y + (direction == Direction.LEFT and -1) + (direction == Direction.RIGHT and 1)

            if x >= self.board_matrix.shape[0] or \
               y >= self.board_matrix.shape[1] or \
               not (self.board_matrix[x, y] == BoardElements.FOOD or self.board_matrix[x, y] == 0):
                hit_object = True
        return counter

    def get_distance_to_food(self, snake, food):
        a = abs(snake[0][0] - food[0])
        b = abs(snake[0][1] - food[1])
        return a + b

    def get_angle_to_food(self, snake, food):
        r = math.atan2(snake[0][0] - food[0], snake[0][1] - food[1])
        degrees = math.degrees(r)
        if degrees < 0:
            degrees += 360.0
        return degrees

    def get_snake_state(self, debug=False):
        state = {}
        state['food_angle'] = self.get_angle_to_food(self.snake, self.food)
        state['food_distance'] = self.get_distance_to_food(self.snake, self.food)
        state['current_direction'] = self.previous_direction.value
        state['object_up'] = self.get_distance_to_object(self.snake, Direction.UP)
        state['object_down'] = self.get_distance_to_object(self.snake, Direction.DOWN)
        state['object_left'] = self.get_distance_to_object(self.snake, Direction.LEFT)
        state['object_right'] = self.get_distance_to_object(self.snake, Direction.RIGHT)
        state['moves_since_last_food'] = self.moves_since_last_food

        # extra parameters for state debugging
        if debug:
            state['head_y'] = self.snake[0][0]
            state['head_x'] = self.snake[0][1]
            state['border_up'] = self.snake[0][0]
            state['border_down'] = self.board_height - 1 - self.snake[0][0]
            state['border_left'] = self.snake[0][1]
            state['border_right'] = self.board_width - 1 - self.snake[0][1]
            state['food_y'] = self.food[0]
            state['food_x'] = self.food[1]
            state['score'] = self.score
            state['life_counter'] = self.life_counter
            state['state'] = self.state.value
            state['snake_length'] = len(self.snake)

        # figure out the best 'next direction'
        if debug:
            up = self.get_distance_to_food([self.move_snake(self.snake, Direction.UP)], self.food)
            down = self.get_distance_to_food([self.move_snake(self.snake, Direction.DOWN)], self.food)
            left = self.get_distance_to_food([self.move_snake(self.snake, Direction.LEFT)], self.food)
            right = self.get_distance_to_food([self.move_snake(self.snake, Direction.RIGHT)], self.food)
            vec = [up, down, left, right]
            state['food_direction_distance'] = vec

        return state

    def print_board(self):
        for i in range(self.board_height):
            for j in range(self.board_width):
                ch = ' '
                m = self.board_matrix[i, j]
                if m == BoardElements.BORDER: ch = '#'
                if m == BoardElements.FOOD: ch = '*'
                if m == BoardElements.SNAKE: ch = 's'
                sys.stdout.write(ch)
            sys.stdout.write('\n')
        sys.stdout.flush()

    def init_curses_board(self):
        curses.initscr()
        self.win = curses.newwin(self.board_width, self.board_height, 0, 0)
        self.win.keypad(1)
        curses.noecho()
        curses.curs_set(0)
        self.win.border(0)
        self.win.nodelay(1)

    def update_board_matrix(self, last_snake_coords=None, last_food_coords=None):
        if last_snake_coords:
            self.board_matrix[last_snake_coords[0], last_snake_coords[1]] = 0.0
        if last_food_coords:
            self.board_matrix[last_food_coords[0], last_food_coords[1]] = 0.0

        self.board_matrix[self.food[0], self.food[1]] = BoardElements.FOOD
        self.board_matrix[self.snake[0][0], self.snake[0][1]] = BoardElements.SNAKE

    def next_move(self, direction: Direction = None, debug: bool = False):
        if direction is None:
            raise ValueError('direction should not be None')

        DEAD_REWARD = -1.0
        FOOD_REWARD = 1.0
        STARVATION_ITERS = self.board_width * 4

        if self.state == SnakeState.DEAD:
            return self.get_snake_state(), DEAD_REWARD, True

        # update previous_direction, grab the previous distance before update
        self.previous_direction = direction
        previous_distance_to_food = self.get_distance_to_food(self.snake, self.food)

        self.snake.insert(0, self.move_snake(self.snake, direction))

        if self.snake[0][0] == 0 or self.snake[0][0] == (self.board_height - 1) or \
           self.snake[0][1] == 0 or self.snake[0][1] == (self.board_width - 1):
            self.state = SnakeState.DEAD
            if debug:
                print('hit the wall')
            return self.get_snake_state(), DEAD_REWARD, True

        # If snake runs over itself
        if self.snake[0] in self.snake[1:]:
            self.state = SnakeState.DEAD
            if debug:
                print(self.snake)
                print('run over itself')
            return self.get_snake_state(), DEAD_REWARD, True

        self.life_counter += 1
        self.moves_since_last_food += 1

        # calculate how far the snake is from food
        def scale(x):
            return math.pow(x, 2)

        def range_scaler(e, lower_bound, upper_bound, min, max):
            numerator = (e - min)
            denominator = (max - min)
            return (upper_bound - lower_bound) * (numerator / denominator) + lower_bound

        distance_to_food = self.get_distance_to_food(self.snake, self.food)
        distance_difference = float(previous_distance_to_food - distance_to_food)
        reward = distance_difference / 10.0
        # reward = scale(1.0 - (distance_to_food / (self.board_width - 2 + self.board_height - 2)))
        # reward = range_scaler(reward, -1.0, 1.0, 0.0, 1.0)

        if self.snake[0] == self.food:
            reward = FOOD_REWARD
            self.update_board_matrix(last_food_coords=(self.food[0], self.food[1]))
            self.food = []
            self.score += 1
            self.moves_since_last_food = 0
            while self.food == []:
                # Calculating next food's coords
                self.food = [randint(1, self.board_height - 2), randint(1, self.board_width - 2)]
                if self.food in self.snake: self.food = []
            if self.win: self.win.addch(self.food[0], self.food[1], '*')
        elif self.moves_since_last_food >= STARVATION_ITERS:
            self.state = SnakeState.DEAD
            if debug: print('starved')
            return self.get_snake_state(), DEAD_REWARD, True
        else:
            last = self.snake.pop()                    # [1] If it does not eat the food, remove tail
            self.update_board_matrix(last_snake_coords=last)
            if self.win: self.win.addch(last[0], last[1], ' ')

        if self.win: self.win.addch(self.snake[0][0], self.snake[0][1], '#')
        return self.get_snake_state(), reward, False

    def onehot_to_move(self, onehot) -> Direction:
        index = onehot.index(max(onehot))
        if index == 0: return Direction.UP
        if index == 1: return Direction.DOWN
        if index == 2: return Direction.LEFT
        if index == 3: return Direction.RIGHT

    def next_move_onehot(self, onehot, debug=False):
        if type(onehot) == torch.Tensor:
            if onehot.is_cuda: onehot = onehot.cpu()
            return self.next_move(self.onehot_to_move(list(onehot.numpy())), debug)
        else:
            return self.next_move(self.onehot_to_move(onehot), debug)

    def next_move_action(self, action, debug=False) -> Direction:
        if action == 0: return self.next_move(Direction.UP, debug)
        if action == 1: return self.next_move(Direction.DOWN, debug)
        if action == 2: return self.next_move(Direction.LEFT, debug)
        if action == 3: return self.next_move(Direction.RIGHT, debug)

