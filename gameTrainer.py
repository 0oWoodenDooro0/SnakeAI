import numpy as np

from direction import Direction
from enviroment import W, H
from food import Food
from point import Point
from snake import Snake


class GameTrainer:
    def __init__(self):
        self.score = 0
        self.steps = 0
        self.snake = Snake()
        self.food = Food()
        self.reset()

    def reset(self):
        self.steps = 0
        self.score = 0
        self.snake.reset()
        self.food.place(self.snake.position)

    def step_by_pygame(self, action):
        self.steps += 1

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

        return reward, game_over, self.score, self.steps

    def step_by_agent(self, action):
        self.steps += 1

        direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        new_dir = self.snake.direction
        if action[0] == 1:
            new_dir = direction[(int(self.snake.direction) + 2) % 4]
        elif action[2] == 1:
            new_dir = direction[int(self.snake.direction) % 4]
        self.snake.move(new_dir)

        reward = 0
        game_over = False
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

        return reward, game_over, self.score, self.steps

    def fitness(self):
        return self.score + self.steps * 0.01 / len(self.snake.position)

    def collision(self, direction, obj):
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
                        return (W - n - 1) / (W - 1)
            elif obj == 'food':
                if pt == self.food.position:
                    return (W - n - 1) / (W - 1)
            elif obj == 'wall':
                if pt.x > W or pt.x < 0 or pt.y > H or pt.y < 0:
                    return n / (W - 1)
            n += 1
        return 0

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
        screen[:, :] = 0.5
        for body in self.snake.position:
            screen[body.y, body.x] = 1
        screen[self.food.position.y, self.food.position.x] = 0
        return screen
