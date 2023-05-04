import torch
import time
import pprint
import os
import click
from snake_simulator import SnakeGame, BoardElements, Direction, SnakeState
from snake_dnn import DQN


def play_game(net, device, board_width, board_height, frame_delay=0.1, step=False):
    game = SnakeGame(board_width=board_width, board_height=board_height)
    total_reward = 0.0
    # bootsrap while loop
    X = net.feature_engineer(game.get_snake_state())
    y_pred = torch.tensor(net(X), device=device)
    snake_state, reward, terminal = game.next_move_onehot(y_pred.tolist())
    while (not terminal):
        total_reward += reward
        clear_command = 'cls' if os.name == 'nt' else 'clear'
        _ = os.system(clear_command)

        game.print_board()
        pprint.pprint(game.get_snake_state(debug=True))
        print('reward: ', reward)
        print('total_reward: ', total_reward)
        action = net.feature_engineer(game.get_snake_state())
        y_pred = torch.tensor(net(action), device=device)
        print(y_pred)
        if step:
            input()

        time.sleep(frame_delay) 
        snake_state, reward, terminal = game.next_move_onehot(y_pred.tolist())

    game.print_board()
    print('prev: {}, attempted: {}'.format(str(game.get_snake_state()['current_direction']),
                                           str(game.onehot_to_move(y_pred.tolist()))))
    print('DEAD')

@click.command()
@click.option('--height', required=True, default=10, help='board height in rows, default=10')
@click.option('--width', required=True, default=30, help='board width in columns, default=30')
@click.option('--frame_delay', required=True, default=0.1, help='delay (in seconds) between game frames, default=0.100')
def main(
    width: int,
    height: int,
    frame_delay: int,
):
    # gpu or cpu 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the current checkpoint
    net: DQN = torch.load('training_state.pt')
    play_game(net, device, board_width=width, board_height=height, frame_delay=frame_delay)


if __name__ == '__main__':
    main()
