import math
import heapq

class MinimaxAStar:
    def __init__(self, player_symbol, max_depth=2, max_nodes=1000):
        """
        Initialize the Minimax algorithm with A* Search principles.
        
        Args:
            player_symbol (str): The symbol of the AI player ('X' or 'O')
            max_depth (int): Maximum depth for the minimax search
            max_nodes (int): Maximum number of nodes to evaluate
        """
        self.player_symbol = player_symbol
        self.opponent_symbol = "O" if player_symbol == "X" else "X"
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.nodes_evaluated = 0  # Counter for performance monitoring
        
    def find_best_move(self, game):
        """
        Find the best move using the minimax algorithm with A* search principles.
        
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
            
        # Use minimax with A* search for other moves
        best_val = -math.inf
        best_move = None
        
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        possible_moves.append((x, y, z))
        
        # Evaluate each move with A* guided minimax
        for x, y, z in possible_moves:
            # Try this move
            game.board[x][y][z] = self.player_symbol
            # Get value from A* guided minimax
            move_val = self.minimax_astar(game, self.max_depth, False)
            # Undo the move
            game.board[x][y][z] = None
            
            # Update best move if needed
            if move_val > best_val:
                best_val = move_val
                best_move = (x, y, z)
        
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        return best_move
    
    def minimax_astar(self, game, depth, is_maximizing):
        """
        The minimax algorithm guided by A* search principles.
        
        Args:
            game: The game object with the current board state
            depth (int): Current depth in the search tree
            is_maximizing (bool): Whether this is a maximizing or minimizing node
            
        Returns:
            int: The evaluation score of the board
        """
        self.nodes_evaluated += 1
        
        # Check if we've evaluated too many nodes
        if self.nodes_evaluated >= self.max_nodes:
            return self.evaluate_board(game)
        
        # Check terminal conditions
        if game.is_winner(self.player_symbol):
            return 1000 + depth  # Prefer winning sooner
        if game.is_winner(self.opponent_symbol):
            return -1000 - depth  # Avoid losing sooner
        if game.is_full() or depth == 0:
            return self.evaluate_board(game)
            
        # Get all possible moves
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if game.board[x][y][z] is None:
                        possible_moves.append((x, y, z))
        
        # Use A* principles to prioritize moves
        move_queue = []
        for x, y, z in possible_moves:
            # Try this move
            if is_maximizing:
                game.board[x][y][z] = self.player_symbol
            else:
                game.board[x][y][z] = self.opponent_symbol
                
            # Calculate heuristic value
            h_value = self.evaluate_board(game)
            
            # For minimizing nodes, we want to prioritize low values
            priority = h_value if is_maximizing else -h_value
            
            # Add to priority queue
            heapq.heappush(move_queue, (-priority, (x, y, z)))
            
            # Undo the move
            game.board[x][y][z] = None
        
        if is_maximizing:
            best_val = -math.inf
            # Process moves in order of decreasing heuristic value
            while move_queue and self.nodes_evaluated < self.max_nodes:
                _, (x, y, z) = heapq.heappop(move_queue)
                game.board[x][y][z] = self.player_symbol
                val = self.minimax_astar(game, depth - 1, False)
                game.board[x][y][z] = None
                best_val = max(best_val, val)
            return best_val
        else:
            best_val = math.inf
            # Process moves in order of increasing heuristic value
            while move_queue and self.nodes_evaluated < self.max_nodes:
                _, (x, y, z) = heapq.heappop(move_queue)
                game.board[x][y][z] = self.opponent_symbol
                val = self.minimax_astar(game, depth - 1, True)
                game.board[x][y][z] = None
                best_val = min(best_val, val)
            return best_val
    
    def evaluate_board(self, game):
        """
        Evaluate the current board state heuristically.
        
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
        
        # Add region control evaluation
        region_score = self.evaluate_regions(game)
        
        # Combine all scores
        return (player_score - opponent_score) * 10 + positional_score + region_score
    
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
                score += player_count * 2  # Player controls region
            elif opponent_count > 0 and player_count == 0:
                score -= opponent_count * 2  # Opponent controls region
        
        return score
    
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
# minimax_astar = MinimaxAStar("X", max_depth=3, max_nodes=2000)
# best_move = minimax_astar.find_best_move(game)
