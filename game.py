import numpy as np
import pygame

from direction import Direction
from enviroment import W, H
from food import Food
from point import Point
from snake import Snake

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
BLOCK_SIZE = 20
WEIGHT = W * BLOCK_SIZE
HEIGHT = H * BLOCK_SIZE
EDGE = 2
SPEED = 20


class Game:
    def __init__(self):
        self.score = 0
        self.steps = 0
        self.display = pygame.display.set_mode((WEIGHT, HEIGHT))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.food = Food()
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
        if self.snake.is_collision():
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
            new_dir = direction[(int(self.snake.direction) + 2) % 4]
        elif action[2] == 1:
            new_dir = direction[int(self.snake.direction) % 4]
        self.snake.move(new_dir)

        game_over = False
        reward = 0
        if self.snake.is_collision():
            game_over = True
            reward = -1
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

    def fitness(self):
        return self.score + self.steps * 0.01 / len(self.snake.position)

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

        pygame.draw.rect(self.display, BLUE1,
                         pygame.Rect(self.snake.head.x * BLOCK_SIZE, self.snake.head.y * BLOCK_SIZE, BLOCK_SIZE,
                                     BLOCK_SIZE))
        for pt in self.snake.position[1:]:
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x * BLOCK_SIZE + EDGE, pt.y * BLOCK_SIZE + EDGE, BLOCK_SIZE - 2 * EDGE,
                                         BLOCK_SIZE - 2 * EDGE))

        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.position.x * BLOCK_SIZE, self.food.position.y * BLOCK_SIZE, BLOCK_SIZE,
                                     BLOCK_SIZE))
        # pygame.draw.line(self.display, BLACK,
        #                  (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                  (WEIGHT, self.snake.head.y + BLOCK_SIZE // 2))
        # pygame.draw.line(self.display, BLACK,
        #                  (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                  (self.snake.head.x + BLOCK_SIZE // 2, HEIGHT))
        # pygame.draw.line(self.display, BLACK,
        #                  (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                  (0, self.snake.head.y + BLOCK_SIZE // 2))
        # pygame.draw.line(self.display, BLACK,
        #                  (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                  (self.snake.head.x + BLOCK_SIZE // 2, 0))
        # pygame.draw.line(self.display, BLACK,
        #                  (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                  (self.snake.head.x + WEIGHT + BLOCK_SIZE // 2, self.snake.head.y - HEIGHT + BLOCK_SIZE // 2))
        # pygame.draw.line(self.display, BLACK,
        #                  (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                  (self.snake.head.x - WEIGHT + BLOCK_SIZE // 2, self.snake.head.y - HEIGHT + BLOCK_SIZE // 2))
        # pygame.draw.line(self.display, BLACK,
        #                  (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                  (self.snake.head.x - WEIGHT + BLOCK_SIZE // 2, self.snake.head.y + HEIGHT + BLOCK_SIZE // 2))
        # pygame.draw.line(self.display, BLACK,
        #                  (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                  (self.snake.head.x + WEIGHT + BLOCK_SIZE // 2, self.snake.head.y + HEIGHT + BLOCK_SIZE // 2))

        # for i in range(8):
        #     food = self.collision(i, 'food', draw=True)
        #     snake = self.collision(i, 'snake', draw=True)
        #     if food:
        #         pygame.draw.line(self.display, RED,
        #                          (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                          (food.x + BLOCK_SIZE // 2, food.y + BLOCK_SIZE // 2))
        #     if snake:
        #         pygame.draw.line(self.display, BLUE2,
        #                          (self.snake.head.x + BLOCK_SIZE // 2, self.snake.head.y + BLOCK_SIZE // 2),
        #                          (snake.x + BLOCK_SIZE // 2, snake.y + BLOCK_SIZE // 2))

        text = font.render('Score: ' + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def collision(self, direction, obj, draw=False):
        pt = self.snake.head
        n = 1
        while not self.snake.is_collision(pt):
            if direction == 0:
                pt = Point(pt.x - 1, pt.y)
            elif direction == 1:
                pt = Point(pt.x, pt.y + 1)
            elif direction == 2:
                pt = Point(pt.x + 1, pt.y)
            elif direction == 3:
                pt = Point(pt.x, pt.y - 1)
            elif direction == 4:
                pt = Point(pt.x - 1, pt.y + 1)
            elif direction == 5:
                pt = Point(pt.x + 1, pt.y + 1)
            elif direction == 6:
                pt = Point(pt.x + 1, pt.y - 1)
            elif direction == 7:
                pt = Point(pt.x - 1, pt.y - 1)
            if obj == 'snake':
                for p in self.snake.position:
                    if p == pt:
                        if draw:
                            return pt
                        return (W - n - 1) / (W - 1)
            elif obj == 'food':
                if pt == self.food.position:
                    if draw:
                        return pt
                    return (W - n - 1) / (W - 1)
            elif obj == 'wall':
                if pt.x > W or pt.x < 0 or pt.y > H or pt.y < 0:
                    return n / (W - 1)
            n += 1
        return 0

    # def draw(self, pt):
    #     pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
    #     pygame.display.flip()

    def check_food(self, direction):
        match direction:
            case Direction.RIGHT:
                if self.food.position.x > self.snake.head.x:
                    return True
            case Direction.LEFT:
                if self.food.position.x < self.snake.head.x:
                    return True
            case Direction.DOWN:
                if self.food.position.y > self.snake.head.y:
                    return True
            case Direction.UP:
                if self.food.position.y < self.snake.head.y:
                    return True
        return False

    def check_danger(self, direction):
        match direction:
            case Direction.RIGHT:
                if self.snake.head.x > W:
                    return True
                for body in self.snake.position:
                    if Point(self.snake.head.x + 1, self.snake.head.y) == body:
                        return True
            case Direction.LEFT:
                if self.snake.head.x < 0:
                    return True
                for body in self.snake.position:
                    if Point(self.snake.head.x - 1, self.snake.head.y) == body:
                        return True
            case Direction.DOWN:
                if self.snake.head.y > H:
                    return True
                for body in self.snake.position:
                    if Point(self.snake.head.x, self.snake.head.y + 1) == body:
                        return True
            case Direction.UP:
                if self.snake.head.y < 0:
                    return True
                for body in self.snake.position:
                    if Point(self.snake.head.x, self.snake.head.y - 1) == body:
                        return True
        return False

    def get_screen(self):
        screen = np.zeros(shape=(H, W), dtype=float)
        screen[:] = 0.5
        for body in self.snake.position:
            screen[body.y][body.x] = 1
        screen[self.food.position.y][self.food.position.x] = 0
        return screen
