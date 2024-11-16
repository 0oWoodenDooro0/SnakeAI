from direction import Direction
from enviroment import W, H
from point import Point


class Snake:
    def __init__(self):
        self.position = None
        self.head = None
        self.direction = None

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(W / 2, H / 2)
        self.position = [self.head]

    def move(self, direction=None):
        if direction is None:
            direction = self.direction
        self.direction = direction
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.LEFT:
            x -= 1
        elif direction == Direction.UP:
            y -= 1
        elif direction == Direction.DOWN:
            y += 1

        self.head = Point(x, y)
        self.position.insert(0, self.head)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > W or pt.x < 0 or pt.y > H or pt.y < 0:
            return True
        if pt in self.position[1:]:
            return True
        return False
