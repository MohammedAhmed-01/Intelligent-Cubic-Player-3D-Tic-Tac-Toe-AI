from tkinter import Tk
from game_logic import CubicGame
from ai_player import AIPlayer
from ui import CubicUI

if __name__ == "__main__":
    root = Tk()
    root.title("3D Tic-Tac-Toe")
    root.geometry("800x700")  # Set a reasonable window size
    
    game = CubicGame()
    ai = AIPlayer("O", algorithm="heuristic")  # Initialize with default algorithm
    app = CubicUI(root, game, ai)
    
    root.mainloop()