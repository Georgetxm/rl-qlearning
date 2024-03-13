import torch
import random
import sys
import numpy as np
from collections import deque  # list optimized for adding and removing items
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness control
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        pass

    def get_state(self, game):
        head = game.snake[0]

        # Creates a point object for each direction around the head
        point_east = Point(head.x + 20, head.y)
        point_west = Point(head.x - 20, head.y)
        point_north = Point(head.x, head.y - 20)
        point_south = Point(head.x, head.y + 20)

        # Boolean values to represent the current game direction
        is_direction_west = game.direction == Direction.WEST
        is_direction_east = game.direction == Direction.EAST
        is_direction_north = game.direction == Direction.NORTH
        is_direction_south = game.direction == Direction.SOUTH

        state = [
            # Relative direction of danger
            # if current game direction matches the direction of the danger point
            (is_direction_east and game.is_collision(point_east)) or
            (is_direction_west and game.is_collision(point_west)) or
            (is_direction_north and game.is_collision(point_north)) or
            (is_direction_south and game.is_collision(point_south)),

            # Danger right
            # if head is facing north, and collision is to the right of head (i.e. absolute direction of east), then it is True)
            (is_direction_north and game.is_collision(point_east)) or
            # if head is facing south, and collision is to the right of head (i.e. absolute direction of west), then it is True)
            (is_direction_south and game.is_collision(point_west)) or
            # if head is facing west, and collision is to the right of head (i.e. absolute direction of north), then it is True)
            (is_direction_west and game.is_collision(point_north)) or
            # if head is facing east, and collision is to the right of head (i.e. absolute direction of south), then it is True)
            (is_direction_east and game.is_collision(point_south)),

            # Danger left
            # if head is facing south, and collision is to the left of head (i.e. absolute direction of east), then it is True)
            (is_direction_south and game.is_collision(point_east)) or
            # if head is facing north, and collision is to the left of head (i.e. absolute direction of west), then it is True)
            (is_direction_north and game.is_collision(point_west)) or
            # if head is facing east, and collision is to the left of head (i.e. absolute direction of north), then it is True)
            (is_direction_east and game.is_collision(point_north)),
            # if head is facing west, and collision is to the left of head (i.e. absolute direction of south), then it is True)
            (is_direction_west and game.is_collision(point_south)) or

            # Boolean to indicate moving direction
            is_direction_west,
            is_direction_east,
            is_direction_north,
            is_direction_south,

            # Food location
            game.food.x < game.head.x,  # food west of head
            game.food.x > game.head.x,  # food east pf head
            game.food.y < game.head.y,  # food north of head
            game.food.y > game.head.y  # food south of head
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        # store 1 tuple of (state, action, reward, next_state, game_over)
        # pop left if MAX_MEMORY reached
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # sample a batch, tuples
        else:
            mini_sample = self.memory

        # Unzip the mini_sample of tuples into 5 lists
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards,
                                next_states, game_overs)

    # Training for 1 game step
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # randome moves: tradeoff exploration vs exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while agent.n_games < 1200:
        # get old state
        state_old = agent.get_state(game)

        # get move based on previous state
        final_move = agent.get_action(state_old)

        # perform move and get new state after final_move
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(
            state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # experience replay
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                if len(sys.argv) > 1:
                    agent.model.save(f"{sys.argv[1]}.pth")
                else:
                    agent.model.save("model_256_1hidden.pth")
            print(
                f'Game {agent.n_games}, Score: {score}, Best Score: {best_score}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

    if len(sys.argv) > 1:
        plot(plot_scores, plot_mean_scores, True, sys.argv[1])
    else:
        plot(plot_scores, plot_mean_scores, True)


if __name__ == '__main__':
    train()
