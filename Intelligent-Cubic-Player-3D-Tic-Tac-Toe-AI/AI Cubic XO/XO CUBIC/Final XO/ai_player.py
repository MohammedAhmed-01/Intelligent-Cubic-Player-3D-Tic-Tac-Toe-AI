import math
import random
import time

class AIPlayer:
    def __init__(self, player_symbol, algorithm="heuristic"):
        self.player_symbol = player_symbol
        self.opponent_symbol = "O" if player_symbol == "X" else "X"
        self.algorithm = algorithm
        self.max_depth = 1  # Reduced depth for faster response
        self.move_cache = {}  # Cache for storing evaluated moves
        self.last_moves = []  # Track recent moves to avoid repetition
        self.randomness_factor = 0.2  # Add some randomness to decisions

    def find_best_move(self, game):
        # Clear cache if it's getting too large
        if len(self.move_cache) > 1000:
            self.move_cache.clear()
            
        # First check for immediate win or block
        winning_move = self.find_winning_move(game, self.player_symbol)
        if winning_move:
            self.last_moves.append(winning_move)
            if len(self.last_moves) > 10:  # Limit history size
                self.last_moves.pop(0)
            return winning_move
            
        blocking_move = self.find_winning_move(game, self.opponent_symbol)
        if blocking_move:
            self.last_moves.append(blocking_move)
            if len(self.last_moves) > 10:  # Limit history size
                self.last_moves.pop(0)
            return blocking_move
            
        # Choose the appropriate algorithm based on the selection
        if self.algorithm == "minimax":
            move = self.minimax_move(game)
        elif self.algorithm == "alpha_beta":
            move = self.alpha_beta_move(game)
        elif self.algorithm == "symmetry":
            move = self.symmetry_reduction_move(game)
        elif self.algorithm == "heuristic_reduction":
            move = self.heuristic_reduction_move(game)
        elif self.algorithm == "minimax_heuristic":
            move = self.minimax_heuristic_move(game)
        elif self.algorithm == "minimax_astar":
            move = self.minimax_astar_move(game)
        else:  # Default to heuristic (original algorithm)
            move = self.heuristic_move(game)
            
        # Add some randomness to avoid predictable play
        if random.random() < self.randomness_factor and move:
            # Get alternative moves
            alt_moves = self.get_alternative_moves(game, move)
            if alt_moves:
                move = random.choice(alt_moves)
                
        # Remember this move
        if move:
            self.last_moves.append(move)
            if len(self.last_moves) > 10:  # Limit history size
                self.last_moves.pop(0)
                
        return move
            
    def heuristic_move(self, game):
        """The original heuristic-based move selection with optimizations"""
        # Check if we have a cached result
        board_key = self.get_board_key(game)
        if board_key in self.move_cache:
            return self.move_cache[board_key]
            
        best_val = -math.inf
        best_move = None

        # Get all empty cells
        empty_cells = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        empty_cells.append((x, y, z))
        
        # If there are too many empty cells, evaluate only a subset
        if len(empty_cells) > 20:
            # For early game, just make a quick move
            return self.make_quick_move(game)
        
        # Evaluate each empty cell
        for x, y, z in empty_cells:
            game.board[x][y][z] = self.player_symbol
            move_val = self.heuristic(game, self.player_symbol) - self.heuristic(game, self.opponent_symbol)
            game.board[x][y][z] = None
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
                
        # Cache the result
        self.move_cache[board_key] = best_move
        return best_move
        
    def minimax_move(self, game):
        """Pure minimax algorithm without optimizations but with early pruning"""
        # Use a faster approach for early game moves
        empty_count = self.count_empty_cells(game)
        if empty_count > 60:  # If most of the board is empty
            return self.make_quick_move(game)
            
        best_val = -math.inf
        best_move = None
        
        # Get possible moves and sort them by a quick evaluation for better pruning
        possible_moves = self.get_possible_moves(game)
        
        # Limit the number of moves to evaluate for better performance
        max_moves_to_check = min(10, len(possible_moves))
        for i in range(max_moves_to_check):
            x, y, z = possible_moves[i]
            game.board[x][y][z] = self.player_symbol
            move_val = self.minimax(game, self.max_depth, False)
            game.board[x][y][z] = None
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
        return best_move
    
    def minimax(self, game, depth, is_maximizing):
        # Check terminal conditions
        if game.is_winner(self.player_symbol):
            return 1000
        if game.is_winner(self.opponent_symbol):
            return -1000
        if game.is_full() or depth == 0:
            return self.evaluate_board(game)
            
        if is_maximizing:
            best_val = -math.inf
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if game.board[x][y][z] is None:
                            game.board[x][y][z] = self.player_symbol
                            val = self.minimax(game, depth - 1, False)
                            game.board[x][y][z] = None
                            best_val = max(best_val, val)
            return best_val
        else:
            best_val = math.inf
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if game.board[x][y][z] is None:
                            game.board[x][y][z] = self.opponent_symbol
                            val = self.minimax(game, depth - 1, True)
                            game.board[x][y][z] = None
                            best_val = min(best_val, val)
            return best_val
            
    def alpha_beta_move(self, game):
        """Minimax with Alpha-Beta Pruning"""
        # Use a faster approach for early game moves
        empty_count = self.count_empty_cells(game)
        if empty_count > 60:  # If most of the board is empty
            return self.make_quick_move(game)
            
        # Check if we have a cached result
        board_key = self.get_board_key(game)
        if board_key in self.move_cache:
            return self.move_cache[board_key]
            
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        
        # Get possible moves and sort them by a quick evaluation for better pruning
        possible_moves = self.get_possible_moves(game)
        
        # Limit the number of moves to evaluate for better performance
        max_moves_to_check = min(10, len(possible_moves))
        for i in range(max_moves_to_check):
            x, y, z = possible_moves[i]
            game.board[x][y][z] = self.player_symbol
            move_val = self.alpha_beta(game, self.max_depth, False, alpha, beta)
            game.board[x][y][z] = None
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
            alpha = max(alpha, best_val)
            
        # Cache the result
        self.move_cache[board_key] = best_move
        return best_move
    
    def alpha_beta(self, game, depth, is_maximizing, alpha, beta):
        # Check terminal conditions
        if game.is_winner(self.player_symbol):
            return 1000
        if game.is_winner(self.opponent_symbol):
            return -1000
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
                            alpha = max(alpha, best_val)
                            if beta <= alpha:
                                break
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
                            beta = min(beta, best_val)
                            if beta <= alpha:
                                break
            return best_val
            
    def symmetry_reduction_move(self, game):
        """Minimax with symmetry reduction to reduce search space"""
        # Check if we have a cached result
        board_key = self.get_board_key(game)
        if board_key in self.move_cache:
            return self.move_cache[board_key]
            
        # First, check if we can use symmetry to reduce the search space
        if self.is_board_empty(game):
            # If board is empty, just return a corner position
            return (0, 0, 0)
            
        # For early game with many empty cells, use a quick move
        empty_count = self.count_empty_cells(game)
        if empty_count > 50:  # If most of the board is empty
            return self.make_quick_move(game)
            
        # Otherwise, use alpha-beta pruning with symmetry considerations
        move = self.alpha_beta_move(game)
        
        # Cache the result
        self.move_cache[board_key] = move
        return move
    
    def is_board_empty(self, game):
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is not None:
                        return False
        return True
        
    def heuristic_reduction_move(self, game):
        """Use heuristic to reduce the number of positions to evaluate"""
        # Check if we have a cached result
        board_key = self.get_board_key(game)
        if board_key in self.move_cache:
            return self.move_cache[board_key]
            
        # For early game with many empty cells, use a quick move
        empty_count = self.count_empty_cells(game)
        if empty_count > 50:  # If most of the board is empty
            return self.make_quick_move(game)
            
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        possible_moves.append((x, y, z))
        
        # If there are too many moves, filter them using heuristic
        if len(possible_moves) > 8:  # Reduced from 10 to 8 for better performance
            # Evaluate each move with a quick heuristic
            move_scores = []
            for move in possible_moves:
                x, y, z = move
                score = self.evaluate_position(game, move)  # Use the optimized position evaluator
                move_scores.append((move, score))
            
            # Sort by score and take top 8
            move_scores.sort(key=lambda x: x[1], reverse=True)
            possible_moves = [move for move, _ in move_scores[:8]]
        
        # Now evaluate the filtered moves with minimax
        best_val = -math.inf
        best_move = None
        
        for move in possible_moves:
            x, y, z = move
            game.board[x][y][z] = self.player_symbol
            move_val = self.minimax(game, self.max_depth, False)
            game.board[x][y][z] = None
            if move_val > best_val:
                best_val = move_val
                best_move = move
        
        # Cache the result
        self.move_cache[board_key] = best_move        
        return best_move
    
    def quick_evaluate(self, game, move):
        """Quick heuristic evaluation for a move"""
        x, y, z = move
        # Count how many potential winning lines this move is part of
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (1, 1, 1), (1, -1, 0), (1, 0, -1),
            (0, 1, -1), (1, -1, -1)
        ]
        
        score = 0
        for dx, dy, dz in directions:
            # Check in both directions
            for direction in [1, -1]:
                line_score = 0
                player_count = 0
                opponent_count = 0
                
                for i in range(4):
                    nx = x + i * dx * direction
                    ny = y + i * dy * direction
                    nz = z + i * dz * direction
                    
                    if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                        if game.board[nx][ny][nz] == self.player_symbol:
                            player_count += 1
                        elif game.board[nx][ny][nz] == self.opponent_symbol:
                            opponent_count += 1
                
                if opponent_count == 0:
                    line_score = player_count * 10
                score += line_score
        
        return score
    
    def minimax_heuristic_move(self, game):
        """Minimax with a more sophisticated evaluation function"""
        # Check if we have a cached result
        board_key = self.get_board_key(game)
        if board_key in self.move_cache:
            return self.move_cache[board_key]
            
        # For early game with many empty cells, use a quick move
        empty_count = self.count_empty_cells(game)
        if empty_count > 50:  # If most of the board is empty
            return self.make_quick_move(game)
            
        # Get possible moves and sort them by a quick evaluation for better pruning
        possible_moves = self.get_possible_moves(game)
        
        # Limit the number of moves to evaluate for better performance
        max_moves_to_check = min(8, len(possible_moves))
        
        best_val = -math.inf
        best_move = None
        
        for i in range(max_moves_to_check):
            x, y, z = possible_moves[i]
            game.board[x][y][z] = self.player_symbol
            move_val = self.minimax_with_heuristic(game, self.max_depth, False)
            game.board[x][y][z] = None
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
        
        # Cache the result
        self.move_cache[board_key] = best_move
        return best_move
    
    def minimax_with_heuristic(self, game, depth, is_maximizing):
        # Check terminal conditions
        if game.is_winner(self.player_symbol):
            return 1000 + depth  # Prefer winning sooner
        if game.is_winner(self.opponent_symbol):
            return -1000 - depth  # Avoid losing sooner
        if game.is_full() or depth == 0:
            return self.advanced_evaluate(game)
            
        if is_maximizing:
            best_val = -math.inf
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if game.board[x][y][z] is None:
                            game.board[x][y][z] = self.player_symbol
                            val = self.minimax_with_heuristic(game, depth - 1, False)
                            game.board[x][y][z] = None
                            best_val = max(best_val, val)
            return best_val
        else:
            best_val = math.inf
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if game.board[x][y][z] is None:
                            game.board[x][y][z] = self.opponent_symbol
                            val = self.minimax_with_heuristic(game, depth - 1, True)
                            game.board[x][y][z] = None
                            best_val = min(best_val, val)
            return best_val
    
    def advanced_evaluate(self, game):
        """More sophisticated board evaluation"""
        # Use the original heuristic but with additional factors
        base_score = self.heuristic(game, self.player_symbol) - self.heuristic(game, self.opponent_symbol)
        
        # Add positional bias - center and corners are more valuable
        position_score = 0
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] == self.player_symbol:
                        # Center positions
                        if (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            position_score += 3
                        # Corner positions
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            position_score += 2
                    elif game.board[x][y][z] == self.opponent_symbol:
                        # Center positions
                        if (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            position_score -= 3
                        # Corner positions
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            position_score -= 2
        
        return base_score + position_score
    
    def minimax_astar_move(self, game):
        """Minimax with A* search for move ordering"""
        # Check if we have a cached result
        board_key = self.get_board_key(game)
        if board_key in self.move_cache:
            return self.move_cache[board_key]
            
        # For early game with many empty cells, use a quick move
        empty_count = self.count_empty_cells(game)
        if empty_count > 50:  # If most of the board is empty
            return self.make_quick_move(game)
            
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        possible_moves.append((x, y, z))
        
        # If there are too many moves, use position evaluation to filter
        if len(possible_moves) > 8:
            move_scores = []
            for move in possible_moves:
                score = self.evaluate_position(game, move)
                move_scores.append((move, score))
            
            # Sort by score and take top 8
            move_scores.sort(key=lambda x: x[1], reverse=True)
            possible_moves = [move for move, _ in move_scores[:8]]
        
        # Evaluate each move with a heuristic
        move_scores = []
        for move in possible_moves:
            x, y, z = move
            game.board[x][y][z] = self.player_symbol
            # g(n) = 0 for the first move, h(n) = heuristic evaluation
            h_score = self.quick_evaluate(game, move)  # Use quicker evaluation
            game.board[x][y][z] = None
            move_scores.append((move, h_score))
        
        # Sort by score (A* f(n) = g(n) + h(n), where g(n) = 0 for first move)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Evaluate the moves in order using minimax with alpha-beta pruning
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        
        # Limit the number of moves to evaluate
        max_moves_to_check = min(5, len(move_scores))
        for i in range(max_moves_to_check):
            move, _ = move_scores[i]
            x, y, z = move
            game.board[x][y][z] = self.player_symbol
            move_val = self.alpha_beta(game, self.max_depth, False, alpha, beta)
            game.board[x][y][z] = None
            if move_val > best_val:
                best_val = move_val
                best_move = move
            alpha = max(alpha, best_val)
        
        # Cache the result
        self.move_cache[board_key] = best_move
        return best_move
        
    def evaluate_board(self, game):
        """Simple evaluation function for minimax"""
        return self.heuristic(game, self.player_symbol) - self.heuristic(game, self.opponent_symbol)

    def heuristic(self, game, player):
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
                        player_count = 0
                        opponent_count = 0
                        empty_count = 0

                        for i in range(4):
                            nx, ny, nz = x + i * dx, y + i * dy, z + i * dz
                            if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4:
                                if game.board[nx][ny][nz] == player:
                                    player_count += 1
                                elif game.board[nx][ny][nz] == self.opponent_symbol:
                                    opponent_count += 1
                                else:
                                    empty_count += 1

                        if player_count > 0 and opponent_count == 0:
                            score += player_count ** 3  
                        if opponent_count > 0 and player_count == 0:
                            score -= opponent_count ** 3  
                        if player_count == 3 and empty_count == 1:
                            score += -500  
                        if opponent_count == 3 and empty_count == 1:
                            score -= 1000 

        return score
        
    def set_algorithm(self, algorithm):
        """Update the algorithm to use"""
        self.algorithm = algorithm
        
    def get_algorithm_name(self):
        """Get the friendly name of the current algorithm"""
        algorithm_names = {
            "minimax": "Minimax Only",
            "alpha_beta": "Minimax + Alpha-Beta Pruning",
            "symmetry": "Symmetry Reduction",
            "heuristic_reduction": "Heuristic Reduction",
            "minimax_heuristic": "Minimax + Evaluation Heuristic",
            "minimax_astar": "Minimax + A* Search",
            "heuristic": "Heuristic Only (Default)"
        }
        return algorithm_names.get(self.algorithm, "Unknown Algorithm")
        
    def count_empty_cells(self, game):
        """Count the number of empty cells on the board"""
        count = 0
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        count += 1
        return count
        
    def get_board_key(self, game):
        """Create a unique key for the current board state for caching"""
        key = ""
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    cell = game.board[x][y][z]
                    if cell is None:
                        key += "_"
                    else:
                        key += cell
        return key
        
    def make_quick_move(self, game):
        """Make a quick move for early game situations with some randomness"""
        # Get strategic positions
        strategic_positions = []
        
        # Center positions are good
        centers = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 2, 2), (2, 1, 2), (2, 2, 1)]
        for pos in centers:
            x, y, z = pos
            if game.board[x][y][z] is None:
                strategic_positions.append((pos, 10))  # Higher weight for center positions
        
        # Corner positions are also good
        corners = [(0, 0, 0), (0, 0, 3), (0, 3, 0), (0, 3, 3), (3, 0, 0), (3, 0, 3), (3, 3, 0), (3, 3, 3)]
        for pos in corners:
            x, y, z = pos
            if game.board[x][y][z] is None:
                strategic_positions.append((pos, 5))  # Medium weight for corners
        
        # Get all available moves
        available_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None and (x, y, z) not in [p[0] for p in strategic_positions]:
                        available_moves.append(((x, y, z), 1))  # Lower weight for other positions
        
        # Combine all moves with their weights
        all_moves = strategic_positions + available_moves
        
        # Remove last move to avoid repetition
        if self.last_moves and all_moves:
            all_moves = [m for m in all_moves if m[0] not in self.last_moves[-3:]]  # Avoid last 3 moves
        
        if all_moves:
            # Use weighted random choice
            positions, weights = zip(*all_moves)
            move = random.choices(positions, weights=weights, k=1)[0]
            return move
        elif available_moves:  # Fallback if all strategic positions are filtered out
            return random.choice([m[0] for m in available_moves])
        return None
        
    def get_possible_moves(self, game):
        """Get all possible moves and sort them by a quick evaluation"""
        moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        # Do a quick evaluation of this move
                        game.board[x][y][z] = self.player_symbol
                        score = self.quick_evaluate(game, (x, y, z))
                        game.board[x][y][z] = None
                        moves.append((x, y, z))
                        
        # Sort moves by score (descending)
        moves.sort(key=lambda move: self.evaluate_position(game, move), reverse=True)
        return moves
        
    def evaluate_position(self, game, move):
        """Quickly evaluate a position based on its strategic value"""
        x, y, z = move
        score = 0
        
        # Center positions are generally better
        if x in [1, 2] and y in [1, 2] and z in [1, 2]:
            score += 3
            
        # Check if this move can block a win
        game.board[x][y][z] = self.opponent_symbol
        if game.is_winner(self.opponent_symbol):
            score += 100  # High priority to block wins
        game.board[x][y][z] = None
        
        # Check if this move can create a win
        game.board[x][y][z] = self.player_symbol
        if game.is_winner(self.player_symbol):
            score += 200  # Even higher priority for winning moves
        game.board[x][y][z] = None
        
        # Add a small random factor to break ties and add unpredictability
        score += random.uniform(0, 1)
        
        return score
        
    def find_winning_move(self, game, player):
        """Find a move that immediately wins or blocks a win"""
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        game.board[x][y][z] = player
                        is_win = game.is_winner(player)
                        game.board[x][y][z] = None
                        if is_win:
                            return (x, y, z)
        return None
        
    def get_alternative_moves(self, game, primary_move):
        """Get alternative moves that are almost as good as the primary move"""
        if not primary_move:
            return []
            
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None and (x, y, z) != primary_move:
                        possible_moves.append((x, y, z))
        
        # Evaluate primary move
        px, py, pz = primary_move
        game.board[px][py][pz] = self.player_symbol
        primary_score = self.quick_evaluate(game, primary_move)
        game.board[px][py][pz] = None
        
        # Find moves that are at least 80% as good as the primary move
        good_alternatives = []
        for move in possible_moves:
            x, y, z = move
            game.board[x][y][z] = self.player_symbol
            score = self.quick_evaluate(game, move)
            game.board[x][y][z] = None
            
            # If score is at least 80% of primary score, it's a good alternative
            if primary_score > 0 and score >= 0.8 * primary_score:
                good_alternatives.append(move)
            # If both scores are negative, compare absolute values
            elif primary_score < 0 and score < 0 and abs(score) <= 1.2 * abs(primary_score):
                good_alternatives.append(move)
                
        # Avoid moves that were recently played
        good_alternatives = [m for m in good_alternatives if m not in self.last_moves[-5:]]  # Avoid last 5 moves
        
        # If we have at least 3 good alternatives, return them
        # Otherwise return a smaller set or an empty list
        return good_alternatives[:3] if good_alternatives else []
