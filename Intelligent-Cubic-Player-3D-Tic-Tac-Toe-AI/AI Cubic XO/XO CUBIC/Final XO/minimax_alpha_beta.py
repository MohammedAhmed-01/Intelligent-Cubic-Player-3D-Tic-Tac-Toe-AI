import math

class MinimaxAlphaBeta:
    def __init__(self, player_symbol, max_depth=2):
        """
        Initialize the Minimax algorithm with Alpha-Beta Pruning.
        
        Args:
            player_symbol (str): The symbol of the AI player ('X' or 'O')
            max_depth (int): Maximum depth for the minimax search
        """
        self.player_symbol = player_symbol
        self.opponent_symbol = "O" if player_symbol == "X" else "X"
        self.max_depth = max_depth
        self.nodes_evaluated = 0  # Counter for performance monitoring
        
    def find_best_move(self, game):
        """
        Find the best move using the minimax algorithm with alpha-beta pruning.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            tuple: The best move as (x, y, z) coordinates
        """
        self.nodes_evaluated = 0  # Reset counter
        
        # First check for immediate win
        winning_move = self.find_winning_move(game, self.player_symbol)
        if winning_move:
            return winning_move
            
        # Then check for immediate block
        blocking_move = self.find_winning_move(game, self.opponent_symbol)
        if blocking_move:
            return blocking_move
            
        # Use minimax with alpha-beta pruning for other moves
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        possible_moves.append((x, y, z))
        
        # Sort moves by a quick evaluation for better pruning
        # This helps alpha-beta pruning work more efficiently
        possible_moves = self.sort_moves(game, possible_moves)
        
        # Evaluate sorted moves
        for x, y, z in possible_moves:
            # Try this move
            game.board[x][y][z] = self.player_symbol
            # Get value from minimax with alpha-beta pruning
            move_val = self.alpha_beta(game, self.max_depth, False, alpha, beta)
            # Undo the move
            game.board[x][y][z] = None
            
            # Update best move if needed
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
            
            # Update alpha
            alpha = max(alpha, best_val)
        
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        return best_move
    
    def alpha_beta(self, game, depth, is_maximizing, alpha, beta):
        """
        The minimax algorithm with alpha-beta pruning.
        
        Args:
            game: The game object with the current board state
            depth (int): Current depth in the search tree
            is_maximizing (bool): Whether this is a maximizing or minimizing node
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            
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
            
        if is_maximizing:
            best_val = -math.inf
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if game.board[x][y][z] is None:
                            game.board[x][y][z] = self.player_symbol
                            val = self.alpha_beta(game, depth - 1, False, alpha, beta)
                            game.board[x][y][z] = None
                            best_val = max(best_val, val)
                            
                            # Alpha-Beta Pruning
                            alpha = max(alpha, best_val)
                            if beta <= alpha:
                                return best_val  # Beta cutoff
            return best_val
        else:
            best_val = math.inf
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if game.board[x][y][z] is None:
                            game.board[x][y][z] = self.opponent_symbol
                            val = self.alpha_beta(game, depth - 1, True, alpha, beta)
                            game.board[x][y][z] = None
                            best_val = min(best_val, val)
                            
                            # Alpha-Beta Pruning
                            beta = min(beta, best_val)
                            if beta <= alpha:
                                return best_val  # Alpha cutoff
            return best_val
    
    def sort_moves(self, game, moves):
        """
        Sort moves by a quick evaluation for better alpha-beta pruning.
        
        Args:
            game: The game object with the current board state
            moves: List of possible moves as (x, y, z) tuples
            
        Returns:
            list: Sorted list of moves
        """
        move_scores = []
        
        for x, y, z in moves:
            # Try this move
            game.board[x][y][z] = self.player_symbol
            # Quick evaluation
            score = self.quick_evaluate(game)
            # Undo the move
            game.board[x][y][z] = None
            
            move_scores.append((score, (x, y, z)))
        
        # Sort by score in descending order
        move_scores.sort(reverse=True)
        
        # Return just the moves
        return [move for _, move in move_scores]
    
    def quick_evaluate(self, game):
        """
        Quick evaluation of the board state for move ordering.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            int: A quick evaluation score
        """
        # Simple heuristic: count lines with only AI pieces
        return self.count_potential_lines(game, self.player_symbol)
    
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
# minimax_ab = MinimaxAlphaBeta("X", max_depth=3)
# best_move = minimax_ab.find_best_move(game)
