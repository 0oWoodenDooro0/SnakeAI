import random
from collections import namedtuple
from enum import Enum


class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4


Point = namedtuple('Point', 'x, y')

W = 20
H = 20
BLOCK_SIZE = 20
WEIGHT = W * BLOCK_SIZE
HEIGHT = H * BLOCK_SIZE


class GameTrainer:
    def __init__(self):
        self.score = None
        self.snake = Snake()
        self.food = Food()
        self.steps = 0
        self.reset()

    def reset(self):
        self.steps = 0
        self.snake.reset()
        self.score = 0
        self.food.place(self.snake.position)

    def step_by_pygame(self, action):
        self.steps += 1

        direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        new_dir = direction[action.index(1)] if self.snake.direction != direction[(action.index(1) + 2) % 4] else self.snake.direction
        self.snake.move(new_dir)

        reward = 0
        game_over = False
        if self.snake.is_collision() or self.steps > 50 * len(self.snake.position):
            game_over = True
            reward = -10
            return reward, game_over, self.score, self.steps

        if self.snake.head == self.food.position:
            self.score += 1
            self.food.place(self.snake.position)
        else:
            self.snake.position.pop()

        return reward, game_over, self.score, self.steps

    def collision(self, direction, obj, draw=False):
        pt = self.snake.head
        n = 1
        while not self.snake.is_collision(pt):
            if direction == 0:
                pt = Point(pt.x - BLOCK_SIZE, pt.y)
            elif direction == 1:
                pt = Point(pt.x, pt.y + BLOCK_SIZE)
            elif direction == 2:
                pt = Point(pt.x + BLOCK_SIZE, pt.y)
            elif direction == 3:
                pt = Point(pt.x, pt.y - BLOCK_SIZE)
            elif direction == 4:
                pt = Point(pt.x - BLOCK_SIZE, pt.y + BLOCK_SIZE)
            elif direction == 5:
                pt = Point(pt.x + BLOCK_SIZE, pt.y + BLOCK_SIZE)
            elif direction == 6:
                pt = Point(pt.x + BLOCK_SIZE, pt.y - BLOCK_SIZE)
            elif direction == 7:
                pt = Point(pt.x - BLOCK_SIZE, pt.y - BLOCK_SIZE)
            if obj == 'snake':
                for p in self.snake.position:
                    if p == pt:
                        if draw:
                            return pt
                        return (BLOCK_SIZE - n - 1) / (BLOCK_SIZE - 1)
            elif obj == 'food':
                if pt == self.food.position:
                    if draw:
                        return pt
                    return (BLOCK_SIZE - n - 1) / (BLOCK_SIZE - 1)
            elif obj == 'wall':
                if pt.x > WEIGHT - BLOCK_SIZE or pt.x < 0 or pt.y > HEIGHT - BLOCK_SIZE or pt.y < 0:
                    return n / (BLOCK_SIZE - 1)
            n += 1
        return 0


class Snake:
    def __init__(self):
        self.position = None
        self.head = None
        self.direction = None

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(WEIGHT / 2, HEIGHT / 2)
        self.position = [self.head]

    def move(self, direction):
        self.direction = direction
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)
        self.position.insert(0, self.head)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > WEIGHT - BLOCK_SIZE or pt.x < 0 or pt.y > HEIGHT - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.position[1:]:
            return True
        return False


class Food:
    def __init__(self):
        self.position = None

    def place(self, snake):
        while True:
            x = random.randint(0, (WEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if Point(x, y) not in snake:
                self.position = Point(x, y)
                break
