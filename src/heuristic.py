# import random
# from settings import *
# from board import is_valid_location, get_next_open_row, get_valid_locations, winning_move

# class HeuristicAgent:
#     def __init__(self, player=AI):
#         self.player = player
#         self.opponent = PLAYER if player == AI else AI
#         center = COLUMNS // 2
#         self.priority = sorted(range(COLUMNS), key=lambda c: abs(c-center))
    
#     def choose_move(self, board):
#         valid_moves = get_valid_locations(board)
#         if not valid_moves:
#             return None
        
#         # 1. Win immediately
#         for col in valid_moves:
#             row = get_next_open_row(board, col)
#             if row is not None:
#                 board_copy = board.copy()
#                 board_copy[row][col] = self.player
#                 win, _ = winning_move(board_copy, self.player)
#                 if win:
#                     return col
        
#         # 2. Block opponent
#         for col in valid_moves:
#             row = get_next_open_row(board, col)
#             if row is not None:
#                 board_copy = board.copy()
#                 board_copy[row][col] = self.opponent
#                 win, _ = winning_move(board_copy, self.opponent)
#                 if win:
#                     return col
        
#         # 3. Center column
#         if COLUMNS//2 in valid_moves:
#             return COLUMNS//2
        
#         # 4. Proximity to center
#         for col in self.priority:
#             if col in valid_moves:
#                 return col
        
#         # Fallback: random move
#         return random.choice(valid_moves)

from settings import *
from board import is_valid_location, get_next_open_row, get_valid_locations, winning_move
import random

class HeuristicAgent:
    def __init__(self, player=AI, alternative_agent=None):
        self.player = player
        self.opponent = PLAYER if player == AI else AI
        center = COLUMNS // 2
        self.priority = sorted(range(COLUMNS), key=lambda c: abs(c-center))
        self.alternative_agent = alternative_agent
        self.use_alternative = False
    
    def decide_strategy(self):
        # Sử dụng alternative agent nếu agent này đóng vai trò PLAYER và có alternative agent
        if self.player == PLAYER and self.alternative_agent is not None:
            self.use_alternative = True
            # Đảm bảo alternative agent cũng biết player ID của nó
            self.alternative_agent.player = self.player
            self.alternative_agent.opponent = self.opponent
        else:
            self.use_alternative = False
    
    def _choose_heuristic_move(self, board):
        valid_moves = get_valid_locations(board)
        if not valid_moves:
            return None
        
        # 1. Win immediately
        for col in valid_moves:
            row = get_next_open_row(board, col)
            if row is not None:
                board_copy = board.copy()
                board_copy[row][col] = self.player
                win, _ = winning_move(board_copy, self.player)
                if win:
                    return col
        
        # 2. Block opponent
        for col in valid_moves:
            row = get_next_open_row(board, col)
            if row is not None:
                board_copy = board.copy()
                board_copy[row][col] = self.opponent
                win, _ = winning_move(board_copy, self.opponent)
                if win:
                    return col
        
        # 3. Center column
        if COLUMNS//2 in valid_moves:
            return COLUMNS//2
        
        # 4. Proximity to center
        for col in self.priority:
            if col in valid_moves:
                return col
        
        # 5. Fallback
        return random.choice(valid_moves)
    
    def choose_move(self, board):
        # Xử lý logic chọn giữa heuristic và alternative
        if self.use_alternative:
            # Ủy quyền cho alternative agent (ví dụ: minimax, neural network...)
            return self.alternative_agent.choose_move(board)
        else:
            # Sử dụng logic heuristic đơn giản
            return self._choose_heuristic_move(board)