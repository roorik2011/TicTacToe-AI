# TicTacToe API 🎮

## Overview 📝
TicTacToe API is a Python-based project that provides an interface for building, training, and using a neural network to play the classic game of Tic-Tac-Toe. The project leverages TensorFlow and Keras to create and manage the model.

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
git clone https://github.com/yourusername/tictactoe-api.git
cd tictactoe-api
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
from tictactoe import TicTacToeAPI

api = TicTacToeAPI()
api.train(epochs=25, batch_size=32)
api.save_model('tictactoe_model.h5')
```

### Loading and Using the Model 🔍
To load a pre-trained model and use it for predictions:
```python
from tictactoe import TicTacToeAPI
import numpy as np

api = TicTacToeAPI()
api.load_model('tictactoe_model.h5')

# Example board state
board = np.array([[1, 0, -1],
                  [0, 1, -1],
                  [0, 0, 0]])

move = api.predict_move(board)
print(f"Predicted move: {move}")
```

### Managing Models 🗂️
List all saved models:
```python
api.list_models()
```

## Examples 📖
Here are a few examples to help you get started.

#### Example 1: Training a Model 📚
```python
from tictactoe import TicTacToeAPI

api = TicTacToeAPI()
api.train(epochs=10, batch_size=32)
api.save_model()
```

#### Example 2: Predicting a Move 🔮
```python
from tictactoe import TicTacToeAPI
import numpy as np

api = TicTacToeAPI()
api.load_model('tictactoe_model.h5')

board = np.array([[1, 0, 0],
                  [0, -1, 1],
                  [0, 0, -1]])

move = api.predict_move(board)
print(f"Predicted move: {move}")
```

### License 📜
This project is licensed under the MIT License. See the LICENSE file for details.
This `README.md` provides a clear and structured guide for users to understand, install, and use your TicTacToe API project.