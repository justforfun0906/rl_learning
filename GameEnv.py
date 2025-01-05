import numpy as np
import random

ACTIONS = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

class GameEnv:
    def __init__(self):
        self.grid_size = 4
        self.reset()
    
    def reset(self):
        # 初始化 4x4 棋盤
        self.board = self.create_grid()
        self.add_new_tile()
        self.add_new_tile()
        self.done = False
        self.score = 0
        
        return self.get_state()
    
    def create_grid(self):
        return [[0] * self.grid_size for _ in range(self.grid_size)]
    
    def add_new_tile(self):
        empty_cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r][c] == 0:
                    empty_cells.append((r, c))
        if not empty_cells:
            return
        r, c = random.choice(empty_cells)
        self.board[r][c] = 4 if random.random() < 0.1 else 2
    
    def get_state(self):
        """
        將 4x4 棋盤壓成一個 np.array (16 維)，或者使用 one-hot / log2 等方式處理。
        為了簡單，這裡直接 flatten 整個棋盤。
        """
        return np.array(self.board).flatten().astype(np.float32)
    
    def compress_line(self, line):
        new_line = [num for num in line if num != 0]
        new_line += [0] * (self.grid_size - len(new_line))
        return new_line

    def merge_line(self, line):
        merged_line = []
        score_gain = 0
        skip = False

        for i in range(self.grid_size):
            if skip:
                skip = False
                continue

            if i < self.grid_size - 1 and line[i] == line[i + 1] and line[i] != 0:
                merged_value = line[i] * 2
                merged_line.append(merged_value)
                score_gain += merged_value
                skip = True
            else:
                merged_line.append(line[i])

        merged_line += [0] * (self.grid_size - len(merged_line))
        return merged_line, score_gain

    def move_up(self):
        transposed_board = [list(row) for row in zip(*self.board)]
        moved, score_gain = self.move_left(transposed_board)
        final_board = [list(row) for row in zip(*moved)]
        return final_board, score_gain

    def move_down(self):
        transposed_board = [list(row) for row in zip(*self.board)]
        moved, score_gain = self.move_right(transposed_board)
        final_board = [list(row) for row in zip(*moved)]
        return final_board, score_gain

    def move_left(self, board=None):
        if board is None:
            board = self.board
        new_board = []
        score_gain = 0
        for row in board:
            compressed = self.compress_line(row)
            merged, gained = self.merge_line(compressed)
            final_line = self.compress_line(merged)
            new_board.append(final_line)
            score_gain += gained
        return new_board, score_gain

    def move_right(self, board=None):
        if board is None:
            board = self.board
        reversed_board = [row[::-1] for row in board]
        moved, score_gain = self.move_left(reversed_board)
        final_board = [row[::-1] for row in moved]
        return final_board, score_gain

    def check_game_over(self, board):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if board[r][c] == 0:
                    return False
                if c < self.grid_size - 1 and board[r][c] == board[r][c + 1]:
                    return False
                if r < self.grid_size - 1 and board[r][c] == board[r + 1][c]:
                    return False
        return True

    def step(self, action):
        old_board = [row[:] for row in self.board]
        
        if action == 0:
            new_board, gained = self.move_up()
        elif action == 1:
            new_board, gained = self.move_down()
        elif action == 2:
            new_board, gained = self.move_left()
        else:
            new_board, gained = self.move_right()
        
        self.board = new_board
        self.score += gained
        
        reward = gained
        
        if self.board != old_board:
            self.add_new_tile()
        
        if self.check_game_over(self.board):
            self.done = True
        
        next_state = self.get_state()
        return next_state, reward, self.done
