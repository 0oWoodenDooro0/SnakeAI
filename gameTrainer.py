from game import Snake, Food, Direction, Point, BLOCK_SIZE, WEIGHT, HEIGHT


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
        if self.snake.is_collision() or self.steps > 20 * pow(len(self.snake.position), 1.2):
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

        game_over = False
        if self.snake.is_collision() or self.steps > 20 * pow(len(self.snake.position), 1.2):
            game_over = True
            fitness = self.fitness()
            return fitness, game_over, self.score, self.steps

        if self.snake.head == self.food.position:
            self.score += 1
            self.food.place(self.snake.position)
        else:
            self.snake.position.pop()

        fitness = self.fitness()

        return fitness, game_over, self.score, self.steps

    def fitness(self):
        return self.score + self.steps * 0.01 / len(self.snake.position)

    def collision(self, direction, obj):
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
                        return (BLOCK_SIZE - n - 1) / (BLOCK_SIZE - 1)
            elif obj == 'food':
                if pt == self.food.position:
                    return (BLOCK_SIZE - n - 1) / (BLOCK_SIZE - 1)
            elif obj == 'wall':
                if pt.x > WEIGHT - BLOCK_SIZE or pt.x < 0 or pt.y > HEIGHT - BLOCK_SIZE or pt.y < 0:
                    return n / (BLOCK_SIZE - 1)
            n += 1
        return 0

    def check_food(self, direction):
        match direction:
            case Direction.RIGHT:
                if self.food.position.x > self.snake.head.x:
                    return True
                else:
                    return False
            case Direction.LEFT:
                if self.food.position.x < self.snake.head.x:
                    return True
                else:
                    return False
            case Direction.DOWN:
                if self.food.position.y > self.snake.head.y:
                    return True
                else:
                    return False
            case Direction.UP:
                if self.food.position.y < self.snake.head.y:
                    return True
                else:
                    return False

    def check_danger(self, direction):
        match direction:
            case Direction.RIGHT:
                if self.snake.head.x > WEIGHT - BLOCK_SIZE: return True
                for body in self.snake.position:
                    if Point(self.snake.head.x + BLOCK_SIZE, self.snake.head.y) == body:
                        return True
            case Direction.LEFT:
                if self.snake.head.x < 0: return True
                for body in self.snake.position:
                    if Point(self.snake.head.x - BLOCK_SIZE, self.snake.head.y) == body:
                        return True
            case Direction.DOWN:
                if self.snake.head.y > HEIGHT - BLOCK_SIZE: return True
                for body in self.snake.position:
                    if Point(self.snake.head.x, self.snake.head.y + BLOCK_SIZE) == body:
                        return True
            case Direction.UP:
                if self.snake.head.y < 0: return True
                for body in self.snake.position:
                    if Point(self.snake.head.x, self.snake.head.y - BLOCK_SIZE) == body:
                        return True
        return False
