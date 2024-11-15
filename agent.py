# import os
# import random
#
# import numpy as np
# import torch
#
# from gameTrainer import Direction, GameTrainer
# from helper import plot
# from model import LinearQNet, DEVICE
#
# POPULATION_SIZE = 100
# MUTATION_RATE = 0.5
# CROSSOVER_RATE = 0.2
# GENERATION_SIZE = 1000
#
#
# class Agent:
#
#     def __init__(self, model):
#         self.model = model
#
#     @staticmethod
#     def get_state(game):
#         state = [
#             # Wall location
#             game.collision(0, 'wall'),
#             game.collision(1, 'wall'),
#             game.collision(2, 'wall'),
#             game.collision(3, 'wall'),
#             game.collision(4, 'wall'),
#             game.collision(5, 'wall'),
#             game.collision(6, 'wall'),
#             game.collision(7, 'wall'),
#
#             # Snake location
#             game.collision(2, 'snake'),
#             game.collision(3, 'snake'),
#             game.collision(0, 'snake'),
#             game.collision(1, 'snake'),
#             game.collision(6, 'snake'),
#             game.collision(7, 'snake'),
#             game.collision(4, 'snake'),
#             game.collision(5, 'snake'),
#
#             # Food location
#             game.collision(2, 'food'),
#             game.collision(3, 'food'),
#             game.collision(0, 'food'),
#             game.collision(1, 'food'),
#             game.collision(6, 'food'),
#             game.collision(7, 'food'),
#             game.collision(4, 'food'),
#             game.collision(5, 'food'),
#
#             game.snake.direction == Direction.RIGHT,
#             game.snake.direction == Direction.UP,
#             game.snake.direction == Direction.LEFT,
#             game.snake.direction == Direction.DOWN
#         ]
#
#         return np.array(state, dtype=float)
#
#     def get_action(self, state):
#         final_move = [0, 0, 0, 0]
#         state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
#         prediction = self.model(state0)
#         move = torch.argmax(prediction).item()
#         final_move[move] = 1
#
#         return final_move
#
#
# def evaluate(score, steps):
#     return score + 0.5 + (0.5 * (score - steps / (score + 1)) / (score + steps / (score + 1)))
#
#
# def snake_ai(model):
#     game = GameTrainer()
#     agent = Agent(model)
#     while True:
#         # get old state
#         state_old = agent.get_state(game)
#
#         # get move
#         final_move = agent.get_action(state_old)
#
#         # perform move and get new state
#         reward, done, score, steps = game.step_by_pygame(final_move)
#
#         if done:
#             fitness = evaluate(score, steps)
#
#             print('Score:', score, 'Steps:', steps, 'Fitness:', fitness)
#
#             return fitness
#
#
# def select_parents(population, fitness):
#     population_size = int(len(population) * CROSSOVER_RATE)
#     population_fitness = list(zip(population, fitness))
#     population_fitness.sort(key=lambda x: x[1], reverse=True)
#     population_fitness = [x for x in population_fitness if x[1] != 0]
#     if population_fitness is None or len(population_fitness) < 2:
#         return [LinearQNet(28, 8, 4) for _ in range(POPULATION_SIZE)]
#     population, fitness = zip(*population_fitness)
#     if len(population) > population_size:
#         elite = list(population)[:population_size]
#     else:
#         elite = list(population)[:len(population)]
#     return elite
#
#
# def crossover(parent1, parent2):
#     # Create a new model with the same architecture as parent1
#     child = LinearQNet(28, 8, 4)
#     child.load_state_dict(parent1.state_dict())
#     # Select a crossover point
#     crossover_point = random.randint(1, len(parent1.state_dict()))
#     # Copy the parameters from parent2 to child
#     for i, (name, param) in enumerate(parent2.state_dict().items()):
#         if i >= crossover_point:
#             child.state_dict()[name].copy_(param)
#     return child
#
#
# def mutation(child):
#     for param in child.parameters():
#         if random.random() < MUTATION_RATE:
#             noise = torch.randn_like(param) * 0.1
#             param.data += noise
#     return child
#
#
# def crossover_and_mutation(parents):
#     children = parents.copy()
#     for _ in range(POPULATION_SIZE - len(parents)):
#         parent1 = parents[random.randint(0, len(parents) - 1)]
#         parent2 = parents[random.randint(0, len(parents) - 1)]
#         child_crossover = crossover(parent1, parent2)
#         child_mutation = mutation(child_crossover)
#         children.append(child_mutation)
#     for idx in range(len(children)):
#         children[idx].save(file_name=f'model{idx + 1}.pth')
#     return children
#
#
# def train():
#     scores = []
#     mean_scores = []
#     generations = 0
#
#     model_folder_path = './model'
#     population = []
#     if os.path.exists(model_folder_path):
#         for i in os.listdir(model_folder_path):
#             model = LinearQNet(28, 8, 4)
#             model.load_state_dict(torch.load(os.path.join(model_folder_path, i)))
#             population.append(model)
#     else:
#         population = [LinearQNet(28, 8, 4) for _ in range(POPULATION_SIZE)]
#
#     while GENERATION_SIZE > generations:
#         # Evaluate the fitness of each individual
#         fitness = [snake_ai(model) for model in population]
#
#         # Select the individuals for crossover
#         parents = select_parents(population, fitness)
#
#         # Crossover and mutation
#         population = crossover_and_mutation(parents)
#
#         fitness.sort(reverse=True)
#         scores.append(np.floor(fitness[0]))
#         mean_scores.append(sum(scores) / len(scores))
#
#         plot(scores, mean_scores)
#         print(f'Generation: {generations} Top_Score: {np.floor(fitness[0])}')
#         generations += 1
#
import json
import os
import pickle
import random
from collections import deque
import keras
from keras import optimizers

import numpy as np
import pandas as pd

from game import Game, Direction
from gameTrainer import GameTrainer
from model import get_model


class Agent:
    def __init__(self, game: Game | GameTrainer):
        self.game = game

    def action(self, action):
        return self.game.step_by_pygame(action.tolist())


class GameState:
    def __init__(self, game, agent: Agent):
        self.game = game
        self.agent = agent

    def get_state(self, actions):
        reward, is_over, score, steps = self.agent.action(actions)
        state = [
            # Wall location
            self.game.collision(0, 'wall'),
            self.game.collision(1, 'wall'),
            self.game.collision(2, 'wall'),
            self.game.collision(3, 'wall'),
            self.game.collision(4, 'wall'),
            self.game.collision(5, 'wall'),
            self.game.collision(6, 'wall'),
            self.game.collision(7, 'wall'),

            # Snake location
            self.game.collision(2, 'snake'),
            self.game.collision(3, 'snake'),
            self.game.collision(0, 'snake'),
            self.game.collision(1, 'snake'),
            self.game.collision(6, 'snake'),
            self.game.collision(7, 'snake'),
            self.game.collision(4, 'snake'),
            self.game.collision(5, 'snake'),

            # Food location
            self.game.collision(2, 'food'),
            self.game.collision(3, 'food'),
            self.game.collision(0, 'food'),
            self.game.collision(1, 'food'),
            self.game.collision(6, 'food'),
            self.game.collision(7, 'food'),
            self.game.collision(4, 'food'),
            self.game.collision(5, 'food'),

            self.game.snake.direction == Direction.RIGHT,
            self.game.snake.direction == Direction.UP,
            self.game.snake.direction == Direction.LEFT,
            self.game.snake.direction == Direction.DOWN
        ]
        return np.reshape(np.array(state, dtype=float), (1, 28)), reward, is_over, score, steps


loss_file_path = 'objects/loss.csv'
scores_file_path = 'objects/scores.csv'
actions_file_path = 'objects/actions.csv'
q_values_file_path = 'objects/q_values.csv'

ACTIONS = 5
GAMMA = 0.99
OBSERVATION = 100
EXPLORE = 100000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 16
FRAMES_PER_ACTION = 1
LEARNING_RATE = 1e-3

loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns=['actions'])
q_values_df = pd.read_csv(q_values_file_path) if os.path.isfile(q_values_file_path) else pd.DataFrame(
    columns=['qvalues'])


def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def init_cache():
    save_obj(INITIAL_EPSILON, "epsilon")
    t = 0
    save_obj(t, "time")
    D = deque()
    save_obj(D, "D")


def train(model: keras.Sequential, game_state: GameState, observe=False):
    D = load_obj("D")
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    s_t, reward, terminal, score, steps = game_state.get_state(do_nothing)
    initial_state = s_t
    if observe:
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        model.load_weights("objects/model.weights.h5")
        adam = optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
    else:
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon")
        model.load_weights("objects/model.weights.h5")
        adam = optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
    t = load_obj("time")
    while True:
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])

        if t % FRAMES_PER_ACTION == 0:
            if random.random() <= epsilon:
                print('Random Action')
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[action_index] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        s_t1, r_t, terminal, score, steps = game_state.get_state(a_t)
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY_SIZE:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH_SIZE)
            inputs = np.zeros((BATCH_SIZE, s_t.shape[1]))
            targets = np.zeros((inputs.shape[0], ACTIONS))

            for i in range(len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                inputs[i:i + 1] = state_t
                targets[i] = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
            scores_df.loc[len(scores_df)] = score

        s_t = initial_state if terminal else s_t1
        if terminal:
            game_state.game.reset()
        t = t + 1
        if t % 1000 == 0:
            print("Model Save")
            model.save_weights("objects/model.weights.h5", overwrite=True)
            save_obj(D, "D")
            save_obj(t, "time")
            save_obj(epsilon, "epsilon")
            loss_df.to_csv(loss_file_path, index=False)
            scores_df.to_csv(scores_file_path, index=False)
            actions_df.to_csv(actions_file_path, index=False)
            q_values_df.to_csv(q_values_file_path, index=False)
            with open("objects/model.json", "w") as file:
                json.dump(model.to_json(), file)
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        fitness = evaluate(score, steps)

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ fitness", fitness,
              "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)


def evaluate(score, steps):
    return score + 0.5 + (0.5 * (score - steps / (score + 1)) / (score + steps / (score + 1)))


def play(show=False, observe=False):
    if show:
        game = Game()
    else:
        game = GameTrainer()
    agent = Agent(game)
    game_state = GameState(game, agent)
    model = get_model()
    train(model, game_state, observe=observe)


if __name__ == "__main__":
    play()
