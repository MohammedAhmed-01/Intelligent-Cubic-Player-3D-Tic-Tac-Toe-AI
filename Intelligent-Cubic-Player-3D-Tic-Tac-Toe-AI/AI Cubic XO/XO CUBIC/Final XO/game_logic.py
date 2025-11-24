class CubicGame:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.board = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
        self.current_player = "X"
        self.winning_positions = []  

    def is_winner(self, player):
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1),
            (1, 1, 1), (1, -1, 0), (1, 0, -1),
            (0, 1, -1), (1, -1, -1)
        ]
        
        self.winning_positions = []  
        
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    for dx, dy, dz in directions:
                        line = self.check_line(player, x, y, z, dx, dy, dz)
                        if line:
                            self.winning_positions = line  
                            return True
        return False

    def check_line(self, player, x, y, z, dx, dy, dz):
        positions = []
        try:
            for i in range(4):
                if (
                    x < 0 or y < 0 or z < 0 or
                    x >= 4 or y >= 4 or z >= 4 or
                    self.board[x][y][z] != player
                ):
                    return None  
                positions.append((x, y, z))
                x += dx
                y += dy
                z += dz
            return positions
        except IndexError:
            return None  

    def is_full(self):
        for layer in self.board:
            for row in layer:
                if None in row:
                    return False
        return True
