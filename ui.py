import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from tictactoe import TicTacToeAI, TicTacToeAPI  # Ensure these classes are in a file named `TicTacToeAI.py`

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

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeUI(root)
    root.mainloop()
