import random
from collections import namedtuple
from enum import Enum

import pygame

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4


Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

W = 20
H = 20
BLOCK_SIZE = 20
WEIGHT = W * BLOCK_SIZE
HEIGHT = H * BLOCK_SIZE
EDGE = 2
SPEED = 20


class Game:
    def __init__(self):
        self.score = None
        self.display = pygame.display.set_mode((WEIGHT, HEIGHT))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if action[0] == 1:
            self.snake.move()
        else:
            direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            new_dir = direction[action.index(1) - 1] if self.snake.direction != direction[
                (action.index(1) + 1) % 4] else self.snake.direction
            self.snake.move(new_dir)

        reward = 0.1
        game_over = False
        if self.snake.is_collision() or self.steps > 100 * len(self.snake.position):
            game_over = True
            reward = 0
            return reward, game_over, self.score, self.steps

        if self.snake.head == self.food.position:
            self.score += 1
            reward = 1
            self.food.place(self.snake.position)
        else:
            self.snake.position.pop()

        self.update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score, self.steps

    def step_by_agent(self, action):
        self.steps += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        new_dir = self.snake.direction
        if action[0] == 1:
            new_dir = direction[(self.snake.direction + 2) % 4]
        elif action[2] == 1:
            new_dir = direction[self.snake.direction % 4]
        self.snake.move(new_dir)

        reward = 0.1
        game_over = False
        if self.snake.is_collision() or self.steps > 100 * len(self.snake.position):
            game_over = True
            reward = 0
            return reward, game_over, self.score, self.steps

        if self.snake.head == self.food.position:
            self.score += 1
            reward = 1
            self.food.place(self.snake.position)
        else:
            self.snake.position.pop()

        self.update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score, self.steps

    def human_step(self):
        direction = self.snake.direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.snake.direction != Direction.RIGHT:
                    direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.snake.direction != Direction.LEFT:
                    direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.snake.direction != Direction.DOWN:
                    direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.snake.direction != Direction.UP:
                    direction = Direction.DOWN
        self.snake.move(direction)

        if self.snake.head == self.food.position:
            self.score += 1
            self.food.place(self.snake.position)
        else:
            self.snake.position.pop()

        if self.snake.is_collision():
            quit()

        self.update_ui()
        self.clock.tick(SPEED)

    def update_ui(self):
        self.display.fill(WHITE)

        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.snake.head.x, self.snake.head.y, BLOCK_SIZE, BLOCK_SIZE))
        for pt in self.snake.position[1:]:
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x + EDGE, pt.y + EDGE, BLOCK_SIZE - 2 * EDGE, BLOCK_SIZE - 2 * EDGE))

        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.position.x, self.food.position.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.line(self.display, BLACK,
                         (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                         (WEIGHT, self.snake.head.y + BLOCK_SIZE // 2))
        pygame.draw.line(self.display, BLACK,
                         (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                         (self.snake.head.x + BLOCK_SIZE // 2, HEIGHT))
        pygame.draw.line(self.display, BLACK,
                         (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                         (0, self.snake.head.y + BLOCK_SIZE // 2))
        pygame.draw.line(self.display, BLACK,
                         (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                         (self.snake.head.x + BLOCK_SIZE // 2, 0))
        pygame.draw.line(self.display, BLACK,
                         (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                         (self.snake.head.x + WEIGHT + BLOCK_SIZE // 2, self.snake.head.y - HEIGHT + BLOCK_SIZE // 2))
        pygame.draw.line(self.display, BLACK,
                         (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                         (self.snake.head.x - WEIGHT + BLOCK_SIZE // 2, self.snake.head.y - HEIGHT + BLOCK_SIZE // 2))
        pygame.draw.line(self.display, BLACK,
                         (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                         (self.snake.head.x - WEIGHT + BLOCK_SIZE // 2, self.snake.head.y + HEIGHT + BLOCK_SIZE // 2))
        pygame.draw.line(self.display, BLACK,
                         (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                         (self.snake.head.x + WEIGHT + BLOCK_SIZE // 2, self.snake.head.y + HEIGHT + BLOCK_SIZE // 2))

        for i in range(8):
            food = self.collision(i, 'food', draw=True)
            snake = self.collision(i, 'snake', draw=True)
            if food:
                pygame.draw.line(self.display, RED,
                                 (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                                 (food.x + BLOCK_SIZE // 2, food.y + BLOCK_SIZE // 2))
            if snake:
                pygame.draw.line(self.display, BLUE2,
                                 (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
                                 (snake.x + BLOCK_SIZE // 2, snake.y + BLOCK_SIZE // 2))

        text = font.render('Score: ' + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

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

    # def draw(self, pt):
    #     pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
    #     pygame.display.flip()


class Snake:
    def __init__(self):
        self.position = None
        self.head = None
        self.direction = None

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(WEIGHT / 2, HEIGHT / 2)
        self.position = [self.head]

    def move(self, direction=None):
        if direction is None:
            direction = self.direction
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
