import os
import random

import numpy as np
import torch

from gameTrainer import Direction, GameTrainer
from helper import plot
from model import LinearQNet, DEVICE

POPULATION_SIZE = 100
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.2


class Agent:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def get_state(game):
        state = [
            # Wall location
            game.collision(0, 'wall'),
            game.collision(1, 'wall'),
            game.collision(2, 'wall'),
            game.collision(3, 'wall'),
            game.collision(4, 'wall'),
            game.collision(5, 'wall'),
            game.collision(6, 'wall'),
            game.collision(7, 'wall'),

            # Snake location
            game.collision(2, 'snake'),
            game.collision(3, 'snake'),
            game.collision(0, 'snake'),
            game.collision(1, 'snake'),
            game.collision(6, 'snake'),
            game.collision(7, 'snake'),
            game.collision(4, 'snake'),
            game.collision(5, 'snake'),

            # Food location
            game.collision(2, 'food'),
            game.collision(3, 'food'),
            game.collision(0, 'food'),
            game.collision(1, 'food'),
            game.collision(6, 'food'),
            game.collision(7, 'food'),
            game.collision(4, 'food'),
            game.collision(5, 'food'),

            game.snake.direction == Direction.RIGHT,
            game.snake.direction == Direction.UP,
            game.snake.direction == Direction.LEFT,
            game.snake.direction == Direction.DOWN
        ]

        return np.array(state, dtype=float)

    def get_action(self, state):
        final_move = [0, 0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


def evaluate(score, steps):
    return score + 0.5 + (0.5 * (score - steps / (score + 1)) / (score + steps / (score + 1)))


def snake_ai(model):
    game = GameTrainer()
    agent = Agent(model)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score, steps = game.step_by_pygame(final_move)

        if done:
            fitness = evaluate(score, steps)

            print('Score:', score, 'Steps:', steps, 'Fitness:', fitness)

            return fitness


def select_parents(population, fitness):
    population_size = int(len(population) * CROSSOVER_RATE)
    population_fitness = list(zip(population, fitness))
    population_fitness.sort(key=lambda x: x[1], reverse=True)
    population_fitness = [x for x in population_fitness if x[1] != 0]
    if population_fitness is None or len(population_fitness) < 2:
        return [LinearQNet(28, 8, 4) for _ in range(POPULATION_SIZE)]
    population, fitness = zip(*population_fitness)
    if len(population) > population_size:
        elite = list(population)[:population_size]
    else:
        elite = list(population)[:len(population)]
    return elite


def crossover(parent1, parent2):
    # Create a new model with the same architecture as parent1
    child = LinearQNet(28, 8, 4)
    child.load_state_dict(parent1.state_dict())
    # Select a crossover point
    crossover_point = random.randint(1, len(parent1.state_dict()))
    # Copy the parameters from parent2 to child
    for i, (name, param) in enumerate(parent2.state_dict().items()):
        if i >= crossover_point:
            child.state_dict()[name].copy_(param)
    return child


def mutation(child):
    for param in child.parameters():
        if random.random() < MUTATION_RATE:
            noise = torch.randn_like(param) * 0.3
            param.data += noise
    return child


def crossover_and_mutation(parents):
    children = parents.copy()
    for _ in range(POPULATION_SIZE - len(parents)):
        parent1 = parents[random.randint(0, len(parents) - 1)]
        parent2 = parents[random.randint(0, len(parents) - 1)]
        child_crossover = crossover(parent1, parent2)
        child_mutation = mutation(child_crossover)
        children.append(child_mutation)
    for idx in range(len(children)):
        children[idx].save(file_name=f'model{idx + 1}.pth')
    return children


def train():
    scores = []
    mean_scores = []
    generations = 0

    model_folder_path = './model'
    population = []
    if os.path.exists(model_folder_path):
        for i in os.listdir(model_folder_path):
            model = LinearQNet(28, 8, 4)
            model.load_state_dict(torch.load(os.path.join(model_folder_path, i)))
            population.append(model)
    else:
        population = [LinearQNet(28, 8, 4) for _ in range(POPULATION_SIZE)]

    while True:
        # Evaluate the fitness of each individual
        fitness = [snake_ai(model) for model in population]

        # Select the individuals for crossover
        parents = select_parents(population, fitness)

        # Crossover and mutation
        population = crossover_and_mutation(parents)

        fitness.sort(reverse=True)
        scores.append(np.floor(fitness[0]))
        mean_scores.append(sum(scores) / len(scores))

        plot(scores, mean_scores)
        print(f'Generation: {generations} Top_Score: {np.floor(fitness[0])}')
        generations += 1


if __name__ == '__main__':
    train()
