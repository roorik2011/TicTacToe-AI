import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
import os
import glob

class TicTacToeAPI:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Flatten(input_shape=(9,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(9, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def generate_training_data(self):
        def check_winner(board, player):
            for i in range(3):
                if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                    return True
            if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
                return True
            return False

        X = []
        y = []
        for _ in range(10000):
            board = np.zeros((3, 3), dtype=int)
            player = 1
            game_moves = []
            for move_count in range(9):
                available_moves = list(zip(*np.where(board == 0)))
                if not available_moves:
                    break
                move = available_moves[np.random.choice(len(available_moves))]
                board[move] = player
                game_moves.append((board.copy(), move[0] * 3 + move[1]))
                if check_winner(board, player):
                    break
                player = -player

            for board_state, move in game_moves:
                X.append(board_state.flatten())
                y.append(move)

        return np.array(X), np.array(y)

    def train(self, epochs=25, batch_size=32):
        X, y = self.generate_training_data()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict_move(self, board):
        pred = self.model.predict(board.flatten().reshape(1, -1))
        move = np.argmax(pred)
        while board.flatten()[move] != 0:
            pred[0][move] = -1
            move = np.argmax(pred)
        return move

    def check_winner(self, board, player):
        for i in range(3):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return True
        if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
            return True
        return False

    def save_model(self, filename=None):
        if filename is None:
            directory = "models"
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_id = len(glob.glob(f"{directory}/save*.h5")) + 1
            filename = f"save{model_id}.h5"
            filepath = os.path.join(directory, filename)
        else:
            filepath = filename
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, model_name):
        directory = "models"
        filepath = os.path.join(directory, model_name)
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")

    def list_models(self, directory="models"):
        if not os.path.exists(directory):
            print("Directory does not exist.")
            return []
        models = glob.glob(f"{directory}/*.h5")
        if not models:
            print("No models found.")
        else:
            print("Available models:")
            for model in models:
                print(model)
        return models
    
    def delete_model(self, model_name):
        directory = "models"
        filepath = os.path.join(directory, model_name)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Model deleted: {filepath}")
        else:
            print(f"Model file not found: {filepath}")
