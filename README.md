# TicTacToe AI + API 🎮

## Overview 📝
TicTacToe AI + API is a Python-based project that provides functionalities for building, training, and using a neural network to play the classic game of Tic-Tac-Toe. The project leverages TensorFlow and Keras to create and manage the model.

## Table of Contents 📚
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Examples](#examples)
5. [License](#license)

## Project Description 🧩
The TicTacToe API is designed to simulate and learn the game of Tic-Tac-Toe using a neural network. The API provides functionalities to:
- Build a neural network model.
- Generate training data by simulating random games.
- Train the model.
- Predict the best move given the current board state.
- Save, load, and manage trained models.

## Installation 🔧
To get started with the TicTacToe API, follow these steps:

### Clone the Repository 📂
```sh
git clone https://github.com/roorik2011/TicTacToe-AI.git
cd TicTacToe-AI
```

### Install Dependencies 📦
Install the required Python libraries using pip:
```bash
pip install numpy tensorflow
```

## Usage 🚀
Here is a brief guide on how to use the TicTacToe API.

### Building and Training the Model 🏗️
To build and train the neural network model:
```python
from tictactoe import TicTacToeAI

ai = TicTacToeAI()
ai.train(epochs=25, batch_size=32)
ai.save_model('tictactoe_model.h5')
```

### Loading and Using the Model 🔍
To load a pre-trained model and use it for predictions:
```python
from tictactoe import TicTacToeAI
import numpy as np

ai = TicTacToeAI()
ai.load_model('tictactoe_model.h5')

# Example board state
board = np.array([[1, 0, -1],
                  [0, 1, -1],
                  [0, 0, 0]])

move = ai.predict_move(board)
print(f"Predicted move: {move}")
```

### Managing Models 🗂️
List all saved models:
```python
ai.list_models()
```
Delete a saved model:
```python
ai.delete_model('tictactoe_model.h5')
```

## Playing the Game with AI 🤖
To play a game where the AI makes moves:
```python
from tictactoe import TicTacToeAI, TicTacToeAPI

ai = TicTacToeAI()
ai.load_model('tictactoe_model.h5')
api = TicTacToeAPI(ai, ai_first=True)

api.reset_board(ai_first=True)  # Reset board with AI starting the game
api.print_board()

# Player makes a move (enter position 0-8)
player_move = int(input("Enter your move (0-8): "))
game_over = api.make_move(player_move)
api.print_board()

# Continue game until there is a winner or a draw
while not game_over:
    game_over = api.ai_move()
    api.print_board()
    if game_over:
        break
    player_move = int(input("Enter your move (0-8): "))
    game_over = api.make_move(player_move)
    api.print_board()
```

### License 📜
This project is licensed under the MIT License. See the LICENSE file for details.
This `README.md` provides a clear and structured guide for users to understand, install, and use your TicTacToe API project.