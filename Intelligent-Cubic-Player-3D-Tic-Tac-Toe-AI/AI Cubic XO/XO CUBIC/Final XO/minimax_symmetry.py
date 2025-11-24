import math
import numpy as np

class MinimaxSymmetry:
    def __init__(self, player_symbol, max_depth=2):
        """
        Initialize the Minimax algorithm with Symmetry Reduction.
        
        Args:
            player_symbol (str): The symbol of the AI player ('X' or 'O')
            max_depth (int): Maximum depth for the minimax search
        """
        self.player_symbol = player_symbol
        self.opponent_symbol = "O" if player_symbol == "X" else "X"
        self.max_depth = max_depth
        self.nodes_evaluated = 0  # Counter for performance monitoring
        self.transposition_table = {}  # Cache for evaluated positions
        
    def find_best_move(self, game):
        """
        Find the best move using the minimax algorithm with symmetry reduction.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            tuple: The best move as (x, y, z) coordinates
        """
        self.nodes_evaluated = 0  # Reset counter
        self.transposition_table = {}  # Clear cache
        
        # First check for immediate win
        winning_move = self.find_winning_move(game, self.player_symbol)
        if winning_move:
            return winning_move
            
        # Then check for immediate block
        blocking_move = self.find_winning_move(game, self.opponent_symbol)
        if blocking_move:
            return blocking_move
            
        # Use minimax with symmetry reduction for other moves
        best_val = -math.inf
        best_move = None
        
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        possible_moves.append((x, y, z))
        
        # Reduce moves by symmetry
        unique_moves = self.reduce_moves_by_symmetry(game, possible_moves)
        
        # Evaluate unique moves
        for x, y, z in unique_moves:
            # Try this move
            game.board[x][y][z] = self.player_symbol
            # Get value from minimax with symmetry reduction
            move_val = self.minimax_symmetry(game, self.max_depth, False)
            # Undo the move
            game.board[x][y][z] = None
            
            # Update best move if needed
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
        
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        print(f"Transposition table size: {len(self.transposition_table)}")
        return best_move
    
    def minimax_symmetry(self, game, depth, is_maximizing):
        """
        The minimax algorithm with symmetry reduction.
        
        Args:
            game: The game object with the current board state
            depth (int): Current depth in the search tree
            is_maximizing (bool): Whether this is a maximizing or minimizing node
            
        Returns:
            int: The evaluation score of the board
        """
        self.nodes_evaluated += 1
        
        # Check terminal conditions
        if game.is_winner(self.player_symbol):
            return 1000 + depth  # Prefer winning sooner
        if game.is_winner(self.opponent_symbol):
            return -1000 - depth  # Avoid losing sooner
        if game.is_full() or depth == 0:
            return self.evaluate_board(game)
        
        # Get canonical board representation
        canonical_board = self.get_canonical_board(game.board)
        board_key = self.board_to_key(canonical_board)
        
        # Check transposition table
        if (board_key, depth, is_maximizing) in self.transposition_table:
            return self.transposition_table[(board_key, depth, is_maximizing)]
            
        if is_maximizing:
            best_val = -math.inf
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if game.board[x][y][z] is None:
                            game.board[x][y][z] = self.player_symbol
                            val = self.minimax_symmetry(game, depth - 1, False)
                            game.board[x][y][z] = None
                            best_val = max(best_val, val)
        else:
            best_val = math.inf
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if game.board[x][y][z] is None:
                            game.board[x][y][z] = self.opponent_symbol
                            val = self.minimax_symmetry(game, depth - 1, True)
                            game.board[x][y][z] = None
                            best_val = min(best_val, val)
        
        # Store in transposition table
        self.transposition_table[(board_key, depth, is_maximizing)] = best_val
        return best_val
    
    def reduce_moves_by_symmetry(self, game, moves):
        """
        Reduce the set of moves by exploiting symmetries.
        
        Args:
            game: The game object with the current board state
            moves: List of possible moves
            
        Returns:
            list: Reduced list of moves
        """
        # For a 4x4x4 cube, we can exploit various symmetries
        # This is a simplified approach - we'll focus on the most obvious symmetries
        
        # If the board is empty or nearly empty, we can reduce to just one corner
        empty_count = sum(1 for x in range(4) for y in range(4) for z in range(4) 
                          if game.board[x][y][z] is None)
        
        if empty_count >= 63:  # Only one move made
            # Just return one corner move
            for move in moves:
                if move == (0, 0, 0):
                    return [move]
            # If corner not available, return center
            for move in moves:
                if move == (1, 1, 1):
                    return [move]
        
        # For partially filled boards, we need more sophisticated symmetry detection
        # This would be a full implementation
        
        # For now, we'll just return all moves
        return moves
    
    def get_canonical_board(self, board):
        """
        Convert the board to its canonical representation.
        
        Args:
            board: The current board state
            
        Returns:
            numpy.ndarray: Canonical board representation
        """
        # Convert board to numpy array for easier manipulation
        np_board = np.zeros((4, 4, 4), dtype=object)
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    np_board[x, y, z] = 0 if board[x][y][z] is None else \
                                       1 if board[x][y][z] == self.player_symbol else 2
        
        # Get all 48 possible orientations of the cube (24 rotations * 2 reflections)
        orientations = []
        
        # Basic orientations (6 faces)
        basic_orientations = [
            np_board,
            np.rot90(np_board, k=1, axes=(0, 1)),  # Rotate around z
            np.rot90(np_board, k=2, axes=(0, 1)),  # Rotate around z twice
            np.rot90(np_board, k=3, axes=(0, 1)),  # Rotate around z three times
            np.rot90(np_board, k=1, axes=(0, 2)),  # Rotate around y
            np.rot90(np_board, k=3, axes=(0, 2)),  # Rotate around y three times
        ]
        
        # For each basic orientation, apply 4 rotations and add to orientations
        for basic in basic_orientations:
            for k in range(4):
                orientations.append(np.rot90(basic, k=k, axes=(1, 2)))
        
        # Add reflections
        for i in range(len(orientations)):
            orientations.append(np.flip(orientations[i], axis=0))
        
        # Find the lexicographically smallest orientation
        canonical = min(orientations, key=lambda x: x.tobytes())
        
        return canonical
    
    def board_to_key(self, board):
        """
        Convert a board to a hashable key.
        
        Args:
            board: The board to convert
            
        Returns:
            bytes: A hashable key representing the board
        """
        return board.tobytes()
    
    def evaluate_board(self, game):
        """
        Evaluate the current board state heuristically.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            int: The evaluation score
        """
        return self.count_potential_lines(game, self.player_symbol) - \
               self.count_potential_lines(game, self.opponent_symbol)
    
    def count_potential_lines(self, game, player):
        """
        Count the number of potential winning lines for a player.
        
        Args:
            game: The game object with the current board state
            player (str): The player symbol to evaluate for
            
        Returns:
            int: Score based on potential winning lines
        """
        score = 0
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (1, 1, 1), (1, -1, 0), (1, 0, -1),
            (0, 1, -1), (1, -1, -1)
        ]
        
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    for dx, dy, dz in directions:
                        line_score = self.evaluate_line(game, player, x, y, z, dx, dy, dz)
                        score += line_score
        
        return score
    
    def evaluate_line(self, game, player, x, y, z, dx, dy, dz):
        """
        Evaluate a potential line for its strategic value.
        
        Args:
            game: The game object with the current board state
            player (str): The player symbol to evaluate for
            x, y, z: Starting coordinates
            dx, dy, dz: Direction vector
            
        Returns:
            int: Score for this line
        """
        opponent = "O" if player == "X" else "X"
        player_count = 0
        opponent_count = 0
        empty_count = 0
        
        try:
            for i in range(4):
                nx, ny, nz = x + i*dx, y + i*dy, z + i*dz
                
                # Check bounds
                if nx < 0 or ny < 0 or nz < 0 or nx >= 4 or ny >= 4 or nz >= 4:
                    return 0  # Line goes out of bounds
                
                cell = game.board[nx][ny][nz]
                if cell == player:
                    player_count += 1
                elif cell == opponent:
                    opponent_count += 1
                else:
                    empty_count += 1
            
            # If opponent has a piece in this line, it's not a potential win
            if opponent_count > 0:
                return 0
            
            # Score based on how many player pieces are in the line
            if player_count == 3 and empty_count == 1:
                return 100  # Almost a win
            elif player_count == 2 and empty_count == 2:
                return 10   # Promising line
            elif player_count == 1 and empty_count == 3:
                return 1    # Potential line
            elif player_count == 0 and empty_count == 4:
                return 0.1  # Empty line
            
            return 0
            
        except IndexError:
            return 0  # Line goes out of bounds
    
    def find_winning_move(self, game, player):
        """
        Find an immediate winning move for the given player.
        
        Args:
            game: The game object with the current board state
            player (str): The player symbol to find a winning move for
            
        Returns:
            tuple or None: The winning move as (x, y, z) coordinates, or None if no winning move exists
        """
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        # Try this move
                        game.board[x][y][z] = player
                        # Check if it's a winning move
                        is_win = game.is_winner(player)
                        # Undo the move
                        game.board[x][y][z] = None
                        
                        if is_win:
                            return (x, y, z)
        
        return None

# Example usage:
# minimax_sym = MinimaxSymmetry("X", max_depth=2)
# best_move = minimax_sym.find_best_move(game)
