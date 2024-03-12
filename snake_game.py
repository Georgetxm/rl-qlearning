import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    EAST = 1
    WEST = 2
    NORTH = 3
    SOUTH = 4


# Point for snake
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (221, 21, 51)
GREEN1 = (59, 124, 44)
GREEN2 = (61, 87, 14)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 10000


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Deep Q Learning')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.WEST

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # array of possible movements
        # [straight, right, left]

        clock_wise = [Direction.EAST, Direction.SOUTH,
                      Direction.WEST, Direction.NORTH]

        # get the current index of the absolute direction
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # no change
            new_abs_direction = clock_wise[index]
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            new_index = (index + 1) % 4
            # right turn west -> south -> east -> north
            new_abs_direction = clock_wise[new_index]
        elif np.array_equal(action, [0, 0, 1]):  # left turn
            new_index = (index - 1) % 4
            # left turn west -> north -> east -> south
            new_abs_direction = clock_wise[new_index]

        self.direction = new_abs_direction

        x = self.head.x
        y = self.head.y

        # change the head position based on the direction
        if self.direction == Direction.EAST:
            x += BLOCK_SIZE
        elif self.direction == Direction.WEST:
            x -= BLOCK_SIZE
        elif self.direction == Direction.SOUTH:
            y += BLOCK_SIZE
        elif self.direction == Direction.NORTH:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
