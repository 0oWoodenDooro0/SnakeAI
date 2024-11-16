import random

from enviroment import W, H
from point import Point


class Food:
    def __init__(self):
        self.position = None

    def place(self, snake):
        while True:
            x = random.randint(0, W)
            y = random.randint(0, H)
            if Point(x, y) not in snake:
                self.position = Point(x, y)
                break
