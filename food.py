import random

from enviroment import W, H
from point import Point


class Food:
    def __init__(self):
        self.position = None

    def place(self, snake):
        while True:
            x = random.randint(0, W - 1)
            y = random.randint(0, H - 1)
            if Point(x, y) not in snake:
                self.position = Point(x, y)
                break
