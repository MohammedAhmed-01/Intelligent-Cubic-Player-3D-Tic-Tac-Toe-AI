import math
import random

class MinimaxHeuristic:
    def __init__(self, player_symbol, max_depth=2, beam_width=10):
        """
        Initialize the Minimax algorithm with Heuristic Reduction.
        
        Args:
            player_symbol (str): The symbol of the AI player ('X' or 'O')
            max_depth (int): Maximum depth for the minimax search
            beam_width (int): Number of top moves to consider at each level
        """
        self.player_symbol = player_symbol
        self.opponent_symbol = "O" if player_symbol == "X" else "X"
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.nodes_evaluated = 0  # Counter for performance monitoring
        self.transposition_table = {}  # Cache for evaluated positions
        
    def find_best_move(self, game):
        """
        Find the best move using the minimax algorithm with heuristic reduction.
        
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
            
        # Use minimax with heuristic reduction for other moves
        best_val = -math.inf
        best_move = None
        
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        possible_moves.append((x, y, z))
        
        # Early game optimization
        empty_count = sum(1 for x in range(4) for y in range(4) for z in range(4) 
                          if game.board[x][y][z] is None)
        
        if empty_count > 60:  # Very early in the game
            # Just make a strategic move
            strategic_moves = [
                (1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1),  # Center and near center
                (0, 0, 0), (0, 0, 3), (0, 3, 0), (3, 0, 0),  # Corners
                (3, 3, 0), (3, 0, 3), (0, 3, 3), (3, 3, 3)
            ]
            for move in strategic_moves:
                if move in possible_moves:
                    return move
            # If no strategic move is available, return a random move
            return random.choice(possible_moves)
        
        # Heuristic pre-evaluation to reduce the search space
        move_scores = []
        for x, y, z in possible_moves:
            # Try this move
            game.board[x][y][z] = self.player_symbol
            # Quick evaluation
            score = self.quick_evaluate(game)
            # Undo the move
            game.board[x][y][z] = None
            
            move_scores.append((score, (x, y, z)))
        
        # Sort by score in descending order
        move_scores.sort(reverse=True)
        
        # Only consider the top N moves (beam search)
        top_moves = [move for _, move in move_scores[:min(self.beam_width, len(move_scores))]]
        
        # Add some randomness to avoid predictable play in equal positions
        if len(top_moves) > 1 and move_scores[0][0] == move_scores[1][0]:
            # If top moves have equal scores, shuffle them
            equal_top_scores = [score for score, _ in move_scores if score == move_scores[0][0]]
            if len(equal_top_scores) > 1:
                top_equal_moves = [move for score, move in move_scores if score == move_scores[0][0]]
                random.shuffle(top_equal_moves)
                top_moves = top_equal_moves + [move for _, move in move_scores 
                                             if move not in top_equal_moves][:self.beam_width]
        
        # Evaluate top moves with minimax
        alpha = -math.inf
        beta = math.inf
        for x, y, z in top_moves:
            # Try this move
            game.board[x][y][z] = self.player_symbol
            # Get value from minimax with alpha-beta pruning
            move_val = self.minimax(game, self.max_depth, False, alpha, beta)
            # Undo the move
            game.board[x][y][z] = None
            
            # Update best move if needed
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
            
            # Update alpha
            alpha = max(alpha, best_val)
        
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        print(f"Transposition table size: {len(self.transposition_table)}")
        return best_move
    
    def minimax(self, game, depth, is_maximizing, alpha, beta):
        """
        The minimax algorithm with alpha-beta pruning and heuristic reduction.
        
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
        
        # Create a key for the transposition table
        board_key = self.board_to_key(game.board)
        
        # Check transposition table
        if (board_key, depth, is_maximizing) in self.transposition_table:
            return self.transposition_table[(board_key, depth, is_maximizing)]
        
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        possible_moves.append((x, y, z))
        
        # Use heuristic to reduce the search space
        if len(possible_moves) > self.beam_width:
            move_scores = []
            for x, y, z in possible_moves:
                # Try this move
                if is_maximizing:
                    game.board[x][y][z] = self.player_symbol
                else:
                    game.board[x][y][z] = self.opponent_symbol
                
                # Quick evaluation
                score = self.quick_evaluate(game) if is_maximizing else -self.quick_evaluate(game)
                
                # Undo the move
                game.board[x][y][z] = None
                
                move_scores.append((score, (x, y, z)))
            
            # Sort by score (descending for maximizing, ascending for minimizing)
            move_scores.sort(reverse=is_maximizing)
            
            # Only consider the top N moves
            possible_moves = [move for _, move in move_scores[:self.beam_width]]
            
        if is_maximizing:
            best_val = -math.inf
            for x, y, z in possible_moves:
                game.board[x][y][z] = self.player_symbol
                val = self.minimax(game, depth - 1, False, alpha, beta)
                game.board[x][y][z] = None
                best_val = max(best_val, val)
                
                # Alpha-Beta Pruning
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break  # Beta cutoff
        else:
            best_val = math.inf
            for x, y, z in possible_moves:
                game.board[x][y][z] = self.opponent_symbol
                val = self.minimax(game, depth - 1, True, alpha, beta)
                game.board[x][y][z] = None
                best_val = min(best_val, val)
                
                # Alpha-Beta Pruning
                beta = min(beta, best_val)
                if beta <= alpha:
                    break  # Alpha cutoff
        
        # Store in transposition table
        self.transposition_table[(board_key, depth, is_maximizing)] = best_val
        return best_val
    
    def board_to_key(self, board):
        """
        Convert a board to a hashable key.
        
        Args:
            board: The board to convert
            
        Returns:
            tuple: A hashable key representing the board
        """
        return tuple(tuple(tuple(cell for cell in row) for row in layer) for layer in board)
    
    def quick_evaluate(self, game):
        """
        Quick evaluation of the board state for move ordering.
        This is a simplified version of the full evaluation.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            int: A quick evaluation score
        """
        # Focus on potential winning lines
        player_score = self.count_potential_lines(game, self.player_symbol)
        opponent_score = self.count_potential_lines(game, self.opponent_symbol)
        
        # Add positional bias - center and corners are more valuable
        positional_score = 0
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] == self.player_symbol:
                        # Center positions
                        if (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            positional_score += 3
                        # Corner positions
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            positional_score += 2
                        # Edge positions
                        elif (x in [0, 3] or y in [0, 3] or z in [0, 3]):
                            positional_score += 1
        
        return player_score - opponent_score + positional_score
    
    def evaluate_board(self, game):
        """
        Comprehensive evaluation of the current board state.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            int: The evaluation score
        """
        # Count potential winning lines
        player_score = self.count_potential_lines(game, self.player_symbol)
        opponent_score = self.count_potential_lines(game, self.opponent_symbol)
        
        # Add positional evaluation
        positional_score = 0
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] == self.player_symbol:
                        # Center positions
                        if (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            positional_score += 3
                        # Corner positions
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            positional_score += 2
                        # Edge positions
                        elif (x in [0, 3] or y in [0, 3] or z in [0, 3]):
                            positional_score += 1
                    elif game.board[x][y][z] == self.opponent_symbol:
                        # Center positions
                        if (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            positional_score -= 3
                        # Corner positions
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            positional_score -= 2
                        # Edge positions
                        elif (x in [0, 3] or y in [0, 3] or z in [0, 3]):
                            positional_score -= 1
        
        # Add control of important regions
        region_score = self.evaluate_regions(game)
        
        return (player_score - opponent_score) * 10 + positional_score + region_score
    
    def evaluate_regions(self, game):
        """
        Evaluate control of important regions on the board.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            int: Score for region control
        """
        # Define important regions
        regions = [
            # Center region
            [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), 
             (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)],
            
            # Corner regions (just one example, there are 8 corners)
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],
            [(3, 0, 0), (3, 0, 1), (3, 1, 0), (2, 0, 0)],
            [(0, 3, 0), (0, 3, 1), (0, 2, 0), (1, 3, 0)],
            [(0, 0, 3), (0, 0, 2), (0, 1, 3), (1, 0, 3)],
            [(3, 3, 0), (3, 3, 1), (3, 2, 0), (2, 3, 0)],
            [(3, 0, 3), (3, 0, 2), (3, 1, 3), (2, 0, 3)],
            [(0, 3, 3), (0, 3, 2), (0, 2, 3), (1, 3, 3)],
            [(3, 3, 3), (3, 3, 2), (3, 2, 3), (2, 3, 3)]
        ]
        
        score = 0
        for region in regions:
            player_count = 0
            opponent_count = 0
            for x, y, z in region:
                if game.board[x][y][z] == self.player_symbol:
                    player_count += 1
                elif game.board[x][y][z] == self.opponent_symbol:
                    opponent_count += 1
            
            # Score based on control of the region
            if player_count > 0 and opponent_count == 0:
                score += player_count * 2  # Player controls region
            elif opponent_count > 0 and player_count == 0:
                score -= opponent_count * 2  # Opponent controls region
        
        return score
    
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
# minimax_heur = MinimaxHeuristic("X", max_depth=3, beam_width=8)
# best_move = minimax_heur.find_best_move(game)
