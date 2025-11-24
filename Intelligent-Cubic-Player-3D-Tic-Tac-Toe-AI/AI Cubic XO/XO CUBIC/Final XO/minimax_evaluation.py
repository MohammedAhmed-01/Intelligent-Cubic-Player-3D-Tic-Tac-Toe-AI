import math

class MinimaxEvaluation:
    def __init__(self, player_symbol, max_depth=2):
        """
        Initialize the Minimax algorithm with advanced evaluation heuristics.
        
        Args:
            player_symbol (str): The symbol of the AI player ('X' or 'O')
            max_depth (int): Maximum depth for the minimax search
        """
        self.player_symbol = player_symbol
        self.opponent_symbol = "O" if player_symbol == "X" else "X"
        self.max_depth = max_depth
        self.nodes_evaluated = 0  # Counter for performance monitoring
        
        # Weights for different evaluation components
        self.weights = {
            'winning_lines': 10.0,
            'blocking_lines': 8.0,
            'center_control': 5.0,
            'corner_control': 3.0,
            'edge_control': 1.0,
            'region_control': 4.0,
            'mobility': 2.0,
            'connectivity': 3.0
        }
        
    def find_best_move(self, game):
        """
        Find the best move using the minimax algorithm with evaluation heuristics.
        
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
        
        # Evaluate all possible moves
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        # Try this move
                        game.board[x][y][z] = self.player_symbol
                        # Get value from minimax
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
        return best_move
    
    def minimax(self, game, depth, is_maximizing, alpha, beta):
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
                            val = self.minimax(game, depth - 1, False, alpha, beta)
                            game.board[x][y][z] = None
                            best_val = max(best_val, val)
                            
                            # Alpha-Beta Pruning
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
                            val = self.minimax(game, depth - 1, True, alpha, beta)
                            game.board[x][y][z] = None
                            best_val = min(best_val, val)
                            
                            # Alpha-Beta Pruning
                            beta = min(beta, best_val)
                            if beta <= alpha:
                                break
            return best_val
    
    def evaluate_board(self, game):
        """
        Advanced evaluation of the current board state using multiple heuristics.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            float: The evaluation score
        """
        # Initialize score components
        score_components = {}
        
        # 1. Evaluate potential winning lines
        player_lines = self.count_potential_lines(game, self.player_symbol)
        opponent_lines = self.count_potential_lines(game, self.opponent_symbol)
        score_components['winning_lines'] = player_lines
        score_components['blocking_lines'] = -opponent_lines
        
        # 2. Evaluate positional control
        position_score = self.evaluate_positions(game)
        score_components['center_control'] = position_score['center']
        score_components['corner_control'] = position_score['corner']
        score_components['edge_control'] = position_score['edge']
        
        # 3. Evaluate region control
        score_components['region_control'] = self.evaluate_regions(game)
        
        # 4. Evaluate mobility (number of available moves after this position)
        score_components['mobility'] = self.evaluate_mobility(game)
        
        # 5. Evaluate connectivity (pieces supporting each other)
        score_components['connectivity'] = self.evaluate_connectivity(game)
        
        # Combine all components with their weights
        final_score = 0
        for component, value in score_components.items():
            final_score += value * self.weights[component]
            
        return final_score
    
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
        empty_positions = []
        
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
                    empty_positions.append((nx, ny, nz))
            
            # If opponent has a piece in this line, it's not a potential win
            if opponent_count > 0:
                return 0
            
            # Score based on how many player pieces are in the line
            if player_count == 3 and empty_count == 1:
                return 100  # Almost a win
            elif player_count == 2 and empty_count == 2:
                # Check if the empty positions are consecutive
                if self.are_positions_consecutive(empty_positions, dx, dy, dz):
                    return 8  # Less valuable than split empty positions
                return 10     # More valuable configuration
            elif player_count == 1 and empty_count == 3:
                return 1    # Potential line
            elif player_count == 0 and empty_count == 4:
                return 0.1  # Empty line
            
            return 0
            
        except IndexError:
            return 0  # Line goes out of bounds
    
    def are_positions_consecutive(self, positions, dx, dy, dz):
        """
        Check if the given positions are consecutive along the direction vector.
        
        Args:
            positions: List of position tuples
            dx, dy, dz: Direction vector
            
        Returns:
            bool: True if positions are consecutive, False otherwise
        """
        if len(positions) < 2:
            return True
            
        # Sort positions along the direction vector
        def projection(pos):
            return pos[0] * dx + pos[1] * dy + pos[2] * dz
            
        sorted_positions = sorted(positions, key=projection)
        
        # Check if they are consecutive
        for i in range(len(sorted_positions) - 1):
            p1 = sorted_positions[i]
            p2 = sorted_positions[i + 1]
            
            # Check if the difference between positions is exactly the direction vector
            if (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]) != (dx, dy, dz):
                return False
                
        return True
    
    def evaluate_positions(self, game):
        """
        Evaluate the control of strategic positions on the board.
        
        Args:
            game: The game object with the current board state
            
        Returns:
            dict: Scores for different position types
        """
        center_score = 0
        corner_score = 0
        edge_score = 0
        
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] == self.player_symbol:
                        # Center positions (most valuable)
                        if (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            center_score += 1
                        # Corner positions (second most valuable)
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            corner_score += 1
                        # Edge positions
                        elif (x in [0, 3] or y in [0, 3] or z in [0, 3]):
                            edge_score += 1
                    elif game.board[x][y][z] == self.opponent_symbol:
                        # Center positions
                        if (x in [1, 2] and y in [1, 2] and z in [1, 2]):
                            center_score -= 1
                        # Corner positions
                        elif (x in [0, 3] and y in [0, 3] and z in [0, 3]):
                            corner_score -= 1
                        # Edge positions
                        elif (x in [0, 3] or y in [0, 3] or z in [0, 3]):
                            edge_score -= 1
        
        return {
            'center': center_score,
            'corner': corner_score,
            'edge': edge_score
        }
    
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

# Example usage:
# minimax_eval = MinimaxEvaluation("X", max_depth=2)
# best_move = minimax_eval.find_best_move(game)
