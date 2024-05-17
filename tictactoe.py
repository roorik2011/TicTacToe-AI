import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten

import tkinter as tk
from tkinter import messagebox, ttk

import os
import glob

class TicTacToeAI:
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

class TicTacToeAPI:
    def __init__(self, ai: TicTacToeAI, ai_first=False):
        self.ai = ai
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = -1 if ai_first else 1  # AI starts if ai_first is True
        if ai_first:
            self.ai_move()

    def reset_board(self, ai_first=False):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = -1 if ai_first else 1
        if ai_first:
            self.ai_move()

    def make_move(self, position):
        if self.board.flatten()[position] == 0:
            self.board[position // 3, position % 3] = self.current_player
            if self.check_winner(self.current_player):
                self.print_board()
                winner = 'AI' if self.current_player == -1 else 'Player'
                print(f"{winner} wins!")
                return True
            if np.all(self.board != 0):
                self.print_board()
                print("It's a draw!")
                return True
            self.current_player = -self.current_player
        else:
            print("Invalid move!")
        return False

    def ai_move(self):
        position = self.ai.predict_move(self.board)
        return self.make_move(position)

    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player or self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return True
        return False

    def print_board(self):
        print(self.board)
        
class TicTacToeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe with Neural Network AI")
        self.ai = TicTacToeAI()
        self.api = None
        
        self.model_list = self.ai.list_models()
        self.selected_model = tk.StringVar(value=self.model_list[0] if self.model_list else '')

        self.create_widgets()
        self.reset_board()

    def create_widgets(self):
        # Left panel for model selection
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Button(left_frame, text="Restart", command=self.reset_board).pack(fill=tk.X, pady=5)
        
        tk.Label(left_frame, text="Select Model:").pack(anchor=tk.W)
        self.model_combo = ttk.Combobox(left_frame, values=self.model_list, textvariable=self.selected_model)
        self.model_combo.pack(fill=tk.X, pady=5)
        tk.Button(left_frame, text="Load Model", command=self.load_model).pack(fill=tk.X, pady=5)
        
        # Center panel for Tic-Tac-Toe game
        center_frame = tk.Frame(self.root)
        center_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(center_frame, text="", font='Helvetica 20 bold', width=5, height=2,
                                command=lambda i=i, j=j: self.player_move(i, j))
                btn.grid(row=i, column=j)
                row.append(btn)
            self.buttons.append(row)
        
        self.status_label = tk.Label(center_frame, text="Player's Turn", font='Helvetica 14')
        self.status_label.grid(row=3, column=0, columnspan=3)

    def load_model(self):
        model_name = self.selected_model.get()
        if model_name:
            self.ai.load_model(model_name)
            self.api = TicTacToeAPI(self.ai)
            self.reset_board()
            self.status_label.config(text="Model Loaded. Player's Turn")
        else:
            messagebox.showwarning("Warning", "No model selected")

    def reset_board(self):
        if self.api:
            self.api.reset_board()
        for row in self.buttons:
            for btn in row:
                btn.config(text="", state=tk.NORMAL)
        self.status_label.config(text="Player's Turn")

    def player_move(self, i, j):
        if not self.api:
            messagebox.showwarning("Warning", "No model loaded")
            return
        
        position = i * 3 + j
        if self.api.make_move(position):
            self.update_board()
            return
        
        self.update_board()
        self.status_label.config(text="AI's Turn")
        self.root.after(500, self.ai_move)

    def ai_move(self):
        if self.api.ai_move():
            self.update_board()
            return
        self.update_board()
        self.status_label.config(text="Player's Turn")

    def update_board(self):
        for i in range(3):
            for j in range(3):
                value = self.api.board[i, j]
                self.buttons[i][j].config(text="X" if value == 1 else "O" if value == -1 else "", state=tk.DISABLED if value != 0 else tk.NORMAL)
        
        if self.api.check_winner(1):
            self.status_label.config(text="Player Wins!")
            self.disable_all_buttons()
        elif self.api.check_winner(-1):
            self.status_label.config(text="AI Wins!")
            self.disable_all_buttons()
        elif not np.any(self.api.board == 0):
            self.status_label.config(text="It's a draw!")
            self.disable_all_buttons()

    def disable_all_buttons(self):
        for row in self.buttons:
            for btn in row:
                btn.config(state=tk.DISABLED)
