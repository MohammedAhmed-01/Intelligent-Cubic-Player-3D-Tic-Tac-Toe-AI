import random

class HeuristicOnly:
    def __init__(self, player_symbol):
        """
        Initialize the Heuristic-only algorithm.
        
        Args:
            player_symbol (str): The symbol of the AI player ('X' or 'O')
        """
        self.player_symbol = player_symbol
        self.opponent_symbol = "O" if player_symbol == "X" else "X"
        self.move_cache = {}  # Cache for storing evaluated moves
        self.randomness_factor = 0.1  # Add some randomness to decisions
        
    def find_best_move(self, game):
        """
        Find the best move using only heuristic evaluation.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            tuple: The best move as (x, y, z) coordinates
        """
        # Clear cache if it's getting too large
        if len(self.move_cache) > 1000:
            self.move_cache.clear()
            
        # Check if we have a cached result for this board state
        board_key = self.board_to_key(game.board)
        if board_key in self.move_cache:
            return self.move_cache[board_key]
            
        # First check for immediate win
        winning_move = self.find_winning_move(game, self.player_symbol)
        if winning_move:
            self.move_cache[board_key] = winning_move
            return winning_move
            
        # Then check for immediate block
        blocking_move = self.find_winning_move(game, self.opponent_symbol)
        if blocking_move:
            self.move_cache[board_key] = blocking_move
            return blocking_move
            
        # Count empty cells to determine game phase
        empty_count = sum(1 for x in range(4) for y in range(4) for z in range(4) 
                          if game.board[x][y][z] is None)
        
        # Early game strategy
        if empty_count > 60:  # Very early in the game
            move = self.early_game_strategy(game)
            self.move_cache[board_key] = move
            return move
            
        # Middle game strategy
        elif empty_count > 30:
            move = self.middle_game_strategy(game)
            self.move_cache[board_key] = move
            return move
            
        # Late game strategy
        else:
            move = self.late_game_strategy(game)
            self.move_cache[board_key] = move
            return move
    
    def early_game_strategy(self, game):
        """
        Strategy for the early game phase.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            tuple: The best move as (x, y, z) coordinates
        """
        # In early game, prioritize center and strategic positions
        strategic_positions = [
            (1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1),  # Center and near center
            (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 2, 2),  # Center and near center
            (0, 0, 0), (0, 0, 3), (0, 3, 0), (3, 0, 0),  # Corners
            (3, 3, 0), (3, 0, 3), (0, 3, 3), (3, 3, 3)   # Corners
        ]
        
        # Check if any strategic position is available
        for pos in strategic_positions:
            x, y, z = pos
            if game.board[x][y][z] is None:
                return pos
                
        # If no strategic position is available, use middle game strategy
        return self.middle_game_strategy(game)
    
    def middle_game_strategy(self, game):
        """
        Strategy for the middle game phase.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            tuple: The best move as (x, y, z) coordinates
        """
        best_val = -float('inf')
        best_move = None
        
        # Get all empty cells
        empty_cells = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        empty_cells.append((x, y, z))
        
        # Evaluate each empty cell
        for x, y, z in empty_cells:
            game.board[x][y][z] = self.player_symbol
            move_val = self.evaluate_position(game)
            game.board[x][y][z] = None
            
            # Add some randomness to avoid predictable play
            if random.random() < self.randomness_factor:
                move_val += random.uniform(-0.1, 0.1)
                
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
                
        return best_move
    
    def late_game_strategy(self, game):
        """
        Strategy for the late game phase.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            tuple: The best move as (x, y, z) coordinates
        """
        best_val = -float('inf')
        best_move = None
        
        # Get all empty cells
        empty_cells = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        empty_cells.append((x, y, z))
        
        # In late game, prioritize moves that create immediate threats
        for x, y, z in empty_cells:
            game.board[x][y][z] = self.player_symbol
            
            # Check if this move creates a threat (one move away from winning)
            threat_count = self.count_threats(game, self.player_symbol)
            
            # Evaluate the overall position
            position_value = self.evaluate_position(game)
            
            # Combine threat count and position value
            move_val = threat_count * 100 + position_value
            
            game.board[x][y][z] = None
            
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
                
        return best_move
    
    def count_threats(self, game, player):
        """
        Count the number of threats (one move away from winning) for a player.
        
        Args:
            game: The game object with the current board state
            player (str): The player symbol to count threats for
            
        Returns:
            int: Number of threats
        """
        threat_count = 0
        
        # Save the current board state
        original_board = [[[game.board[x][y][z] for z in range(4)] for y in range(4)] for x in range(4)]
        
        # Check each empty cell
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        # Try this move
                        game.board[x][y][z] = player
                        # Check if it's a winning move
                        if game.is_winner(player):
                            threat_count += 1
                        # Undo the move
                        game.board[x][y][z] = None
        
        # Restore the original board state
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    game.board[x][y][z] = original_board[x][y][z]
                    
        return threat_count
    
    def evaluate_position(self, game):
        """
        Comprehensive evaluation of the current board state.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            float: The evaluation score
        """
        # 1. Evaluate potential winning lines
        line_score = self.evaluate_lines(game)
        
        # 2. Evaluate positional control
        position_score = self.evaluate_positions(game)
        
        # 3. Evaluate region control
        region_score = self.evaluate_regions(game)
        
        # 4. Evaluate mobility
        mobility_score = self.evaluate_mobility(game)
        
        # 5. Evaluate connectivity
        connectivity_score = self.evaluate_connectivity(game)
        
        # Combine all scores with appropriate weights
        total_score = (
            line_score * 10.0 +
            position_score * 5.0 +
            region_score * 3.0 +
            mobility_score * 2.0 +
            connectivity_score * 1.5
        )
        
        return total_score
    
    def evaluate_lines(self, game):
        """
        Evaluate potential winning lines for both players.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            float: Score based on line evaluation
        """
        player_score = self.count_potential_lines(game, self.player_symbol)
        opponent_score = self.count_potential_lines(game, self.opponent_symbol)
        
        return player_score - opponent_score
    
    def count_potential_lines(self, game, player):
        """
        Count the number of potential winning lines for a player.
        
        Args:
            game: The game object with the current board state
            player (str): The player symbol to evaluate for
            
        Returns:
            float: Score based on potential winning lines
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
            float: Score for this line
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
    
    def evaluate_positions(self, game):
        """
        Evaluate the control of strategic positions on the board.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            float: Score for position control
        """
        score = 0
        
        # Define position values
        position_values = {
            'center': 3.0,  # Center positions
            'near_center': 2.0,  # Positions adjacent to center
            'corner': 2.5,  # Corner positions
            'edge': 1.0,  # Edge positions
            'other': 0.5  # Other positions
        }
        
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] == self.player_symbol:
                        # Center positions
                        if x == 1 and y == 1 and z == 1 or x == 2 and y == 2 and z == 2 or \
                           x == 1 and y == 2 and z == 2 or x == 2 and y == 1 and z == 1:
                            score += position_values['center']
                        # Near center positions
                        elif (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            score += position_values['near_center']
                        # Corner positions
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            score += position_values['corner']
                        # Edge positions
                        elif (x in [0, 3] or y in [0, 3] or z in [0, 3]):
                            score += position_values['edge']
                        # Other positions
                        else:
                            score += position_values['other']
                    elif game.board[x][y][z] == self.opponent_symbol:
                        # Center positions
                        if x == 1 and y == 1 and z == 1 or x == 2 and y == 2 and z == 2 or \
                           x == 1 and y == 2 and z == 2 or x == 2 and y == 1 and z == 1:
                            score -= position_values['center']
                        # Near center positions
                        elif (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            score -= position_values['near_center']
                        # Corner positions
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            score -= position_values['corner']
                        # Edge positions
                        elif (x in [0, 3] or y in [0, 3] or z in [0, 3]):
                            score -= position_values['edge']
                        # Other positions
                        else:
                            score -= position_values['other']
        
        return score
    
    def evaluate_regions(self, game):
        """
        Evaluate control of important regions on the board.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            float: Score for region control
        """
        # Define important regions
        regions = [
            # Center region
            [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), 
             (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)],
            
            # Corner regions
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
                score += player_count * (1 + 0.1 * player_count)  # Bonus for multiple pieces
            elif opponent_count > 0 and player_count == 0:
                score -= opponent_count * (1 + 0.1 * opponent_count)
        
        return score
    
    def evaluate_mobility(self, game):
        """
        Evaluate the mobility (available moves) for both players.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            float: Mobility score
        """
        # Count empty cells that are adjacent to player's pieces
        player_mobility = 0
        opponent_mobility = 0
        
        # Define adjacency (including diagonals)
        adjacency = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
        
        # Track which empty cells have been counted
        counted_cells = set()
        
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] == self.player_symbol:
                        # Check adjacent cells
                        for dx, dy, dz in adjacency:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and
                                game.board[nx][ny][nz] is None and
                                (nx, ny, nz) not in counted_cells):
                                player_mobility += 1
                                counted_cells.add((nx, ny, nz))
                    elif game.board[x][y][z] == self.opponent_symbol:
                        # Check adjacent cells
                        for dx, dy, dz in adjacency:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and
                                game.board[nx][ny][nz] is None and
                                (nx, ny, nz) not in counted_cells):
                                opponent_mobility += 1
                                counted_cells.add((nx, ny, nz))
        
        return player_mobility - opponent_mobility
    
    def evaluate_connectivity(self, game):
        """
        Evaluate how well the pieces are connected to each other.
        Connected pieces are more likely to form winning lines.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            float: Connectivity score
        """
        player_connectivity = 0
        opponent_connectivity = 0
        
        # Define adjacency (including diagonals)
        adjacency = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
        
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] == self.player_symbol:
                        # Count adjacent player pieces
                        for dx, dy, dz in adjacency:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and
                                game.board[nx][ny][nz] == self.player_symbol):
                                player_connectivity += 1
                    elif game.board[x][y][z] == self.opponent_symbol:
                        # Count adjacent opponent pieces
                        for dx, dy, dz in adjacency:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and
                                game.board[nx][ny][nz] == self.opponent_symbol):
                                opponent_connectivity += 1
        
        # Normalize by dividing by 2 since each connection is counted twice
        return (player_connectivity - opponent_connectivity) / 2
    
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
    
    def board_to_key(self, board):
        """
        Convert a board to a hashable key for caching.
        
        Args:
            board: The board to convert
            
        Returns:
            tuple: A hashable key representing the board
        """
        return tuple(tuple(tuple(cell for cell in row) for row in layer) for layer in board)

# Example usage:
# heuristic_ai = HeuristicOnly("X")
# best_move = heuristic_ai.find_best_move(game)
