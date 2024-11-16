# import os
#
# import torch
# import torch.nn as nn
#
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# class LinearQNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size).to(device=DEVICE)
#         self.fc2 = nn.Linear(hidden_size, output_size).to(device=DEVICE)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
#
#     def save(self, file_name='model.pth'):
#         model_folder_path = './model'
#         if not os.path.exists(model_folder_path):
#             os.makedirs(model_folder_path)
#         file_name = os.path.join(model_folder_path, file_name)
#         torch.save(self.state_dict(), file_name)
import os

import keras
from keras import layers
from keras import optimizers

model_file_path = 'objects/model.weights.h5'


def get_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 2)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=3))
    adam = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=adam)
    if not os.path.isfile(model_file_path):
        model.save_weights(model_file_path)
    print("Model Create Successfully")
    return model
