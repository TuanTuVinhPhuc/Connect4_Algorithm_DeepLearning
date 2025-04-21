import numpy as np
import math
import random
import time
from functools import lru_cache

from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
from board import is_valid_location, get_next_open_row, get_valid_locations, drop_piece, winning_move

C_PARAM = math.sqrt(2) 
MIN_SIMULATIONS = 30000
TIME_LIMIT = 10

# Sử dụng lru_cache để lưu kết quả đánh giá cửa sổ
@lru_cache(maxsize=1024)
def evaluate_window_cached(window_tuple, piece):
    """Phiên bản lưu cache của evaluate_window"""
    window = list(window_tuple)
    opp_piece = PLAYER if piece == AI else AI
    
    piece_count = window.count(piece)
    empty_count = window.count(EMPTY)
    opp_count = window.count(opp_piece)
    
    if piece_count == 4:
        return 100
    elif piece_count == 3 and empty_count == 1:
        return 5
    elif piece_count == 2 and empty_count == 2:
        return 2
    
    if opp_count == 3 and empty_count == 1:
        return -6
    elif opp_count == 2 and empty_count == 2:
        return -3
    
    return 0

class MCTSNode:
    def __init__(self, board, parent=None, move=None, player=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  
        self.player = player  
        self.children = []  
        self.wins = 0 
        self.visits = 0 
        self.untried_moves = get_valid_locations(board)  
        self.terminal = False  
        self.total_value = 0.0
        
        if winning_move(board, AI)[0] or winning_move(board, PLAYER)[0] or len(self.untried_moves) == 0:
            self.terminal = True
    
    def uct_select_child(self):
        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        
        log_total_visits = math.log(self.visits)
        
        return max(self.children, key=lambda c: c.get_uct_value(log_total_visits))
            
    def get_uct_value(self, log_parent_visits):
        if self.visits == 0:
            return float('inf')
        
        avg_value = self.total_value / self.visits
        
        exploration = C_PARAM * math.sqrt(log_parent_visits / self.visits)
        
        center_bias = 0
        if self.move is not None:
            center_bias = (1.0 - abs(self.move - COLUMNS // 2) / (COLUMNS // 2)) * 0.1
        
        return avg_value + exploration + center_bias
    
    def expand(self):
        if not self.untried_moves or self.terminal:
            return None
        
        # Ưu tiên các nước đi ở giữa bàn cờ
        center_col = COLUMNS // 2
        weighted_moves = []
        for move in self.untried_moves:
            weight = 1.0 + (1.0 - abs(move - center_col) / center_col) * 2.0
            weighted_moves.append((move, weight))
        
        moves, weights = zip(*weighted_moves)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Random bias về hướng cột middle.
        move = np.random.choice(moves, p=normalized_weights)
        
        self.untried_moves.remove(move)
        
        child_board = self.board.copy()
        row = get_next_open_row(child_board, move)
        if row is not None:
            drop_piece(child_board, row, move, self.player)
            
            next_player = PLAYER if self.player == AI else AI
            child_node = MCTSNode(child_board, self, move, next_player)
            self.children.append(child_node)
            
            return child_node
        
        return None
    
    def update(self, result):
        self.visits += 1
        self.total_value += result
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        return self.terminal
    
    def get_best_move(self):
        if not self.children:
            return None
        
        visited_children = [child for child in self.children if child.visits > 0]
        if not visited_children:
            return None
            
        best_child = max(visited_children, key=lambda c: c.visits)
        
        high_win_rate_children = [child for child in visited_children 
                                 if child.visits >= max(100, best_child.visits * 0.3) and 
                                 child.total_value / child.visits >= 0.9]
        
        if high_win_rate_children:
            return max(high_win_rate_children, 
                      key=lambda c: (c.total_value / c.visits) + 0.0001 * c.visits).move
        
        return int(best_child.move)

def evaluate_window(window, piece):
    """Wrapper cho hàm cached"""
    return evaluate_window_cached(tuple(window), piece)

# Bộ nhớ cache cho evaluate_board
board_eval_cache = {}

def evaluate_board(board, piece):
    """Đánh giá bàn cờ với lưu trữ đệm"""
    # Chuyển thành tuple để có thể làm key cho cache
    if isinstance(board, np.ndarray):
        board_key = tuple(map(tuple, board))
    else:
        board_key = tuple(map(tuple, board))
        
    cache_key = (board_key, piece)
    
    # Kiểm tra trong cache
    if cache_key in board_eval_cache:
        return board_eval_cache[cache_key]
    
    # Tính toán nếu không có trong cache
    score = 0

    center_array = [board[r][COLUMNS//2] for r in range(ROWS)]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    # Kiểm tra theo hàng ngang
    for r in range(ROWS):
        for c in range(COLUMNS-3):
            window = [board[r][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Kiểm tra theo hàng dọc
    for c in range(COLUMNS):
        for r in range(ROWS-3):
            window = [board[r+i][c] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Kiểm tra theo đường chéo /
    for r in range(ROWS-3):
        for c in range(COLUMNS-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Kiểm tra theo đường chéo \
    for r in range(3, ROWS):
        for c in range(COLUMNS-3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Lưu kết quả vào cache
    board_eval_cache[cache_key] = score
    
    # Xóa bớt cache nếu quá lớn (chỉ giữ 1000 mục)
    if len(board_eval_cache) > 1000:
        # Xóa 20% các mục cũ nhất
        keys_to_delete = list(board_eval_cache.keys())[:200]
        for key in keys_to_delete:
            del board_eval_cache[key]
            
    return score

# Cache cho hàm detect_two_way_win
two_way_win_cache = {}

def detect_two_way_win(board, player):
    """Kiểm tra two-way win với lưu trữ đệm"""
    # Tạo key cho cache
    if isinstance(board, np.ndarray):
        board_key = tuple(map(tuple, board))
    else:
        board_key = tuple(map(tuple, board))
        
    cache_key = (board_key, player)
    
    # Kiểm tra cache
    if cache_key in two_way_win_cache:
        return two_way_win_cache[cache_key]
    
    valid_moves = get_valid_locations(board)
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue

        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, player)
        
        win_threats = 0
        win_columns = []
        
        for next_move in valid_moves:
            if next_move == move:  
                continue
                
            next_row = get_next_open_row(b_copy, next_move)
            if next_row is None:
                continue
                
            temp_b_copy = [r[:] for r in b_copy]
            drop_piece(temp_b_copy, next_row, next_move, player)
            
            if winning_move(temp_b_copy, player)[0]:
                win_threats += 1
                win_columns.append(next_move)
                
        if win_threats >= 2:
            result = (move, win_threats)
            two_way_win_cache[cache_key] = result
            return result
    
    result = (None, 0)
    two_way_win_cache[cache_key] = result
    return result

# Cache cho find_forced_win
forced_win_cache = {}

def find_forced_win(board, player, depth):
    """Tìm forced win với lưu trữ đệm"""
    if depth <= 0:
        return None
    
    # Tạo key cho cache
    if isinstance(board, np.ndarray):
        board_key = tuple(map(tuple, board))
    else:
        board_key = tuple(map(tuple, board))
        
    cache_key = (board_key, player, depth)
    
    # Kiểm tra cache
    if cache_key in forced_win_cache:
        return forced_win_cache[cache_key]
        
    valid_moves = get_valid_locations(board)
    opponent = PLAYER if player == AI else AI
    
    # Kiểm tra chiến thắng ngay lập tức
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, player)
        
        if winning_move(b_copy, player)[0]:
            forced_win_cache[cache_key] = move
            return move
            
    # Kiểm tra two-way win
    two_way_move, _ = detect_two_way_win(board, player)
    if two_way_move is not None:
        forced_win_cache[cache_key] = two_way_move
        return two_way_move

    # Kiểm tra forced win sâu hơn
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, player)
    
        opponent_valid_moves = get_valid_locations(b_copy)
        opponent_can_prevent_win = False
        
        for opponent_move in opponent_valid_moves:
            opponent_row = get_next_open_row(b_copy, opponent_move)
            if opponent_row is None:
                continue
                
            opponent_b_copy = [r[:] for r in b_copy]
            drop_piece(opponent_b_copy, opponent_row, opponent_move, opponent)
            
            if winning_move(opponent_b_copy, opponent)[0]:
                opponent_can_prevent_win = True
                break
                
            player_win_next = find_forced_win(opponent_b_copy, player, depth-1)
            
            if player_win_next is None:
                opponent_can_prevent_win = True
                break
                
        if not opponent_can_prevent_win:
            forced_win_cache[cache_key] = move
            return move
    
    # Không tìm thấy forced win
    forced_win_cache[cache_key] = None
    return None

def rollout_policy(board, player):
    """Policy cho rollout giai đoạn mô phỏng"""
    valid_moves = get_valid_locations(board)
    if not valid_moves:
        return None

    # Kiểm tra chiến thắng ngay lập tức
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, player)
        
        if winning_move(b_copy, player)[0]:
            return move
    
    # Kiểm tra chặn đối thủ thắng
    opponent = PLAYER if player == AI else AI
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, opponent)
        
        if winning_move(b_copy, opponent)[0]:
            return move

    # Kiểm tra two-way win
    two_way_move, _ = detect_two_way_win(board, player)
    if two_way_move is not None:
        return two_way_move

    opponent_two_way, _ = detect_two_way_win(board, opponent)
    if opponent_two_way is not None:
        return opponent_two_way
    
    # Kiểm tra forced win
    forced_win = find_forced_win(board, player, 4)
    if forced_win is not None:
        return forced_win
    
    # Tính điểm cho các nước đi
    move_scores = []

    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, player)
        
        score = evaluate_board(b_copy, player)
        
        # Ưu tiên các cột ở giữa
        center_preference = (1.0 - abs(move - COLUMNS // 2) / (COLUMNS // 2)) * 2
        score += center_preference
        
        # Tìm 3 quân liên tiếp với 1 ô trống
        for c in range(COLUMNS-3):
            for r in range(ROWS):
                window = [b_copy[r][c+i] for i in range(4)]
                if window.count(player) == 3 and window.count(EMPTY) == 1:
                    score += 3

        move_scores.append((move, score))
            
    if move_scores:
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_moves = move_scores[:min(3, len(move_scores))]
        
        if len(top_moves) > 1:
            if top_moves[0][1] > 0 and top_moves[0][1] >= top_moves[1][1] * 1.5:
                return top_moves[0][0]
            
            total_score = sum(score for _, score in top_moves)
            if total_score != 0:  
                weights = [score / total_score for _, score in top_moves]
                moves = [move for move, _ in top_moves]
                
                # Random theo xác suất, bias về phía move có điểm cao hơn. 
                return random.choices(moves, weights=weights, k=1)[0]
            
        return top_moves[0][0] 
    
    # Ưu tiên cột giữa nếu có thể
    if COLUMNS // 2 in valid_moves:
        return COLUMNS // 2

    return random.choice(valid_moves)
    
def rollout(board, player):
    """Mô phỏng một ván chơi từ trạng thái board"""
    current_board = board.copy()
    opponent = PLAYER if player == AI else AI
    current_player = player
    depth = 0
    max_depth = ROWS * COLUMNS  
    
    while depth < max_depth:
        if winning_move(current_board, player)[0]:
            return 1.0  # Thắng
        elif winning_move(current_board, opponent)[0]:
            return 0.0  # Thua
        
        valid_moves = get_valid_locations(current_board)
        if not valid_moves:
            return 0.5  # Hòa
        
        move = rollout_policy(current_board, current_player)
            
        row = get_next_open_row(current_board, move)
        if row is not None:
            drop_piece(current_board, row, move, current_player)
            
            current_player = PLAYER if current_player == AI else AI
            depth += 1
        else:
            break
    
    # Không có kết quả rõ ràng, đánh giá trạng thái bàn cờ
    player_score = evaluate_board(current_board, player)
    opponent_score = evaluate_board(current_board, opponent)
    
    total_score = abs(opponent_score) + abs(player_score)
    if total_score == 0:
        return 0.5
    
    normalized_score = (player_score + total_score) / (2 * total_score)
    return min(max(normalized_score, 0.0), 1.0)  

def mcts_search(board, player, simulations=MIN_SIMULATIONS, time_limit=TIME_LIMIT):
    """Tìm kiếm nước đi tốt nhất bằng MCTS"""
    # Reset cache cho mỗi lần tìm kiếm mới
    global board_eval_cache, two_way_win_cache, forced_win_cache
    board_eval_cache = {}
    two_way_win_cache = {}
    forced_win_cache = {}
    
    if not isinstance(board, np.ndarray):
        board = np.array(board)
    
    # Thực hiện kiểm tra thắng/thua nhanh trước khi bắt đầu MCTS
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is not None:
                board_copy = board.copy()
                drop_piece(board_copy, row, col, player)
                if winning_move(board_copy, player)[0]:
                    return col
    
    opponent = PLAYER if player == AI else AI
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is not None:
                board_copy = board.copy()
                drop_piece(board_copy, row, col, opponent)
                if winning_move(board_copy, opponent)[0]:
                    return col
                    
    two_way_move, _ = detect_two_way_win(board.tolist(), player)
    if two_way_move is not None:
        return two_way_move

    opponent_two_way, _ = detect_two_way_win(board.tolist(), opponent)
    if opponent_two_way is not None:
        return opponent_two_way

    forced_win_move = find_forced_win(board.tolist(), player, 4)
    if forced_win_move is not None:
        return forced_win_move
    
    # Ưu tiên cột giữa nếu là bàn cờ trống
    is_empty_board = True
    for r in range(ROWS):
        for c in range(COLUMNS):
            if board[r, c] != EMPTY:
                is_empty_board = False
                break
        if not is_empty_board:
            break
            
    if is_empty_board:
        return COLUMNS // 2
    
    # Ưu tiên cột giữa nếu còn trống
    if board[ROWS-1, COLUMNS//2] == EMPTY:
        return COLUMNS // 2
    
    # Bắt đầu MCTS
    root = MCTSNode(board, player=player)
    
    if len(root.untried_moves) == 1:
        return root.untried_moves[0]

    start_time = time.time()
    sim_count = 0
    
    max_time = time_limit * 0.9
    
    pieces_count = np.count_nonzero(board != 0)
    time_factor = min(1.0, pieces_count / (ROWS * COLUMNS * 0.7))
    
    actual_max_time = max_time * (0.9 + 0.1 * time_factor)
    
    while ((time.time() - start_time) < actual_max_time) and (sim_count < simulations):
        node = root
        
        # Chọn
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.uct_select_child()
        
        # Mở rộng
        if not node.is_terminal():
            new_node = node.expand()
            if new_node:
                node = new_node

        # Mô phỏng
        result = rollout(node.board, node.player)
  
        # Cập nhật ngược
        while node:
            if node.player == player:
                node.update(1 - result)
            else:
                node.update(result)
            node = node.parent
        
        sim_count += 1
        
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        visits_threshold = min(100, sim_count // 3) 
        
        if best_child and best_child.visits > visits_threshold:
            win_rate = best_child.total_value / best_child.visits

            if win_rate > 0.95 and sim_count > simulations:
                break
            
            if sim_count > simulations and win_rate > 0.8:
                second_best = sorted(root.children, key=lambda c: c.visits)[-2] if len(root.children) >= 2 else None
                if second_best and best_child.visits > second_best.visits * 1.5:
                    break
    
    elapsed_time = time.time() - start_time
    print(f"MCTS: {sim_count} mô phỏng trong {elapsed_time:.3f}s ({elapsed_time/time_limit*100:.1f}% thời gian)")

    total_visits = sum(child.visits for child in root.children)

    if root.children:
        print(f"Thống kê mô phỏng theo cột:")
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
            if child.visits > 0:
                win_rate = child.total_value / child.visits
                win_percent = win_rate * 100
                visit_percent = (child.visits / total_visits) * 100 if total_visits > 0 else 0
                print(f"  Cột {child.move}: {child.visits} lần thăm ({visit_percent:.1f}% tổng số), tỷ lệ thắng: {win_percent:.1f}%")

    remaining_time = max_time - elapsed_time
    if remaining_time > 0:
        print(f"Chờ thêm {remaining_time:.3f}s để đảm bảo thời gian đồng bộ")
        time.sleep(max(0, remaining_time))
    
    return int(root.get_best_move())

def main():
    matrix = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0],
        [0, 0, 2, 1, 1, 0, 0]
    ]

    move = mcts_search(matrix, PLAYER)
    print(move)

if __name__ == "__main__":
    main()




# import numpy as np
# import math
# import random
# import time
# from functools import lru_cache

# from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
# from board import is_valid_location, get_next_open_row, get_valid_locations, drop_piece, winning_move

# C_PARAM = math.sqrt(2) 
# MIN_SIMULATIONS = 50000
# TIME_LIMIT = 10

# # Sử dụng lru_cache để lưu kết quả đánh giá cửa sổ
# @lru_cache(maxsize=4096)
# def evaluate_window_cached(window_tuple, piece):
#     """Phiên bản lưu cache của evaluate_window"""
#     window = list(window_tuple)
#     opp_piece = PLAYER if piece == AI else AI
    
#     piece_count = window.count(piece)
#     empty_count = window.count(EMPTY)
#     opp_count = window.count(opp_piece)
    
#     if piece_count == 4:
#         return 100
#     elif piece_count == 3 and empty_count == 1:
#         return 5
#     elif piece_count == 2 and empty_count == 2:
#         return 2
    
#     if opp_count == 3 and empty_count == 1:
#         return -6
#     elif opp_count == 2 and empty_count == 2:
#         return -3
    
#     return 0

# def evaluate_window(window, piece):
#     """Wrapper cho hàm cached"""
#     return evaluate_window_cached(tuple(window), piece)

# # Bộ nhớ cache cho evaluate_board
# board_eval_cache = {}

# def evaluate_board(board, piece):
#     """Đánh giá bàn cờ với lưu trữ đệm"""
#     # Sử dụng hash nhanh hơn
#     if isinstance(board, np.ndarray):
#         board_key = hash(board.tobytes())
#     else:
#         board_key = hash(str(board))
        
#     cache_key = (board_key, piece)
    
#     # Kiểm tra trong cache
#     if cache_key in board_eval_cache:
#         return board_eval_cache[cache_key]
    
#     # Tính toán nếu không có trong cache
#     score = 0

#     center_array = [board[r][COLUMNS//2] for r in range(ROWS)]
#     center_count = center_array.count(piece)
#     score += center_count * 3
    
#     # Kiểm tra theo hàng ngang
#     for r in range(ROWS):
#         for c in range(COLUMNS-3):
#             window = [board[r][c+i] for i in range(4)]
#             score += evaluate_window(window, piece)
    
#     # Kiểm tra theo hàng dọc
#     for c in range(COLUMNS):
#         for r in range(ROWS-3):
#             window = [board[r+i][c] for i in range(4)]
#             score += evaluate_window(window, piece)
    
#     # Kiểm tra theo đường chéo /
#     for r in range(ROWS-3):
#         for c in range(COLUMNS-3):
#             window = [board[r+i][c+i] for i in range(4)]
#             score += evaluate_window(window, piece)
    
#     # Kiểm tra theo đường chéo \
#     for r in range(3, ROWS):
#         for c in range(COLUMNS-3):
#             window = [board[r-i][c+i] for i in range(4)]
#             score += evaluate_window(window, piece)
    
#     # Lưu kết quả vào cache
#     board_eval_cache[cache_key] = score
    
#     # Xóa bớt cache nếu quá lớn (chỉ giữ 1000 mục)
#     if len(board_eval_cache) > 1000:
#         # Xóa 20% các mục cũ nhất
#         keys_to_delete = list(board_eval_cache.keys())[:200]
#         for key in keys_to_delete:
#             del board_eval_cache[key]
            
#     return score

# def detect_two_way_win(board, player):
#     """Kiểm tra two-way win"""
#     valid_moves = get_valid_locations(board)
#     for move in valid_moves:
#         row = get_next_open_row(board, move)
#         if row is None:
#             continue

#         # Sử dụng numpy để hiệu quả hơn
#         b_copy = np.array(board).copy() if isinstance(board, np.ndarray) else np.array([r[:] for r in board])
#         drop_piece(b_copy, row, move, player)
        
#         win_threats = 0
        
#         for next_move in valid_moves:
#             if next_move == move:  
#                 continue
                
#             next_row = get_next_open_row(b_copy, next_move)
#             if next_row is None:
#                 continue
                
#             temp_b_copy = b_copy.copy()
#             drop_piece(temp_b_copy, next_row, next_move, player)
            
#             if winning_move(temp_b_copy, player)[0]:
#                 win_threats += 1
#                 if win_threats >= 2:  # Dừng ngay khi tìm thấy 2 đường thắng
#                     return move, win_threats
                
#     return None, 0

# def find_forced_win(board, player, depth):
#     """Tìm forced win"""
#     if depth <= 0:
#         return None
        
#     valid_moves = get_valid_locations(board)
#     opponent = PLAYER if player == AI else AI
    
#     # Kiểm tra thắng ngay lập tức
#     for move in valid_moves:
#         row = get_next_open_row(board, move)
#         if row is None:
#             continue
            
#         b_copy = np.array(board).copy() if isinstance(board, np.ndarray) else np.array([r[:] for r in board])
#         drop_piece(b_copy, row, move, player)
        
#         if winning_move(b_copy, player)[0]:
#             return move
            
#     # Kiểm tra two-way win
#     two_way_move, _ = detect_two_way_win(board, player)
#     if two_way_move is not None:
#         return two_way_move

#     # Giới hạn độ sâu tìm kiếm để tiết kiệm thời gian
#     if depth <= 2:
#         for move in valid_moves:
#             row = get_next_open_row(board, move)
#             if row is None:
#                 continue
                
#             b_copy = np.array(board).copy() if isinstance(board, np.ndarray) else np.array([r[:] for r in board])
#             drop_piece(b_copy, row, move, player)
        
#             opponent_valid_moves = get_valid_locations(b_copy)
#             opponent_can_prevent_win = False
            
#             for opponent_move in opponent_valid_moves:
#                 opponent_row = get_next_open_row(b_copy, opponent_move)
#                 if opponent_row is None:
#                     continue
                    
#                 opponent_b_copy = b_copy.copy()
#                 drop_piece(opponent_b_copy, opponent_row, opponent_move, opponent)
                
#                 if winning_move(opponent_b_copy, opponent)[0]:
#                     opponent_can_prevent_win = True
#                     break
                    
#                 player_win_next = find_forced_win(opponent_b_copy, player, depth-1)
                
#                 if player_win_next is None:
#                     opponent_can_prevent_win = True
#                     break
                    
#             if not opponent_can_prevent_win:
#                 return move
                
#     return None

# # ===== ROLLOUT PHẦN LIGHT =====
# def light_rollout_policy(board, player):
#     """Policy nhẹ cho rollout ở midgame: 
#     - Thắng/thua ngay lập tức
#     - Ưu tiên cột giữa
#     - Ngẫu nhiên có trọng số
#     """
#     valid_moves = get_valid_locations(board)
#     if not valid_moves:
#         return None

#     # Kiểm tra thắng ngay lập tức
#     for move in valid_moves:
#         row = get_next_open_row(board, move)
#         if row is None:
#             continue
            
#         b_copy = board.copy()
#         drop_piece(b_copy, row, move, player)
        
#         if winning_move(b_copy, player)[0]:
#             return move
            
#     # Kiểm tra chặn đối thủ thắng
#     opponent = PLAYER if player == AI else AI
#     for move in valid_moves:
#         row = get_next_open_row(board, move)
#         if row is None:
#             continue
            
#         b_copy = board.copy()
#         drop_piece(b_copy, row, move, opponent)
        
#         if winning_move(b_copy, opponent)[0]:
#             return move
    
#     # Ưu tiên cột giữa với trọng số ngẫu nhiên
#     center_weights = [1 + (1 - abs(move - COLUMNS//2)/(COLUMNS//2)) for move in valid_moves]
#     total_weight = sum(center_weights)
#     norm_weights = [w/total_weight for w in center_weights]
    
#     # Trả về nước đi ngẫu nhiên có trọng số
#     return random.choices(valid_moves, weights=norm_weights, k=1)[0]

# # ===== ROLLOUT PHẦN HEAVY =====
# def heavy_rollout_policy(board, player):
#     """Policy nặng cho rollout ở endgame"""
#     valid_moves = get_valid_locations(board)
#     if not valid_moves:
#         return None

#     # Kiểm tra thắng ngay lập tức
#     for move in valid_moves:
#         row = get_next_open_row(board, move)
#         if row is None:
#             continue
            
#         b_copy = board.copy()
#         drop_piece(b_copy, row, move, player)
        
#         if winning_move(b_copy, player)[0]:
#             return move
            
#     # Kiểm tra chặn đối thủ thắng
#     opponent = PLAYER if player == AI else AI
#     for move in valid_moves:
#         row = get_next_open_row(board, move)
#         if row is None:
#             continue
            
#         b_copy = board.copy()
#         drop_piece(b_copy, row, move, opponent)
        
#         if winning_move(b_copy, opponent)[0]:
#             return move

#     # Kiểm tra two-way win
#     two_way_move, _ = detect_two_way_win(board, player)
#     if two_way_move is not None:
#         return two_way_move

#     opponent_two_way, _ = detect_two_way_win(board, opponent)
#     if opponent_two_way is not None:
#         return opponent_two_way
    
#     # Tính điểm cho các nước đi
#     move_scores = []

#     for move in valid_moves:
#         row = get_next_open_row(board, move)
#         if row is None:
#             continue
            
#         b_copy = board.copy()
#         drop_piece(b_copy, row, move, player)
        
#         score = evaluate_board(b_copy, player)
        
#         # Ưu tiên các cột ở giữa
#         center_preference = (1.0 - abs(move - COLUMNS // 2) / (COLUMNS // 2)) * 2
#         score += center_preference

#         move_scores.append((move, score))
            
#     if move_scores:
#         move_scores.sort(key=lambda x: x[1], reverse=True)
        
#         top_moves = move_scores[:min(3, len(move_scores))]
        
#         if len(top_moves) > 1:
#             if top_moves[0][1] > 0 and top_moves[0][1] >= top_moves[1][1] * 1.5:
#                 return top_moves[0][0]
            
#             total_score = sum(score for _, score in top_moves)
#             if total_score <= 0:
#                 # Nếu tổng điểm <= 0, ưu tiên cột giữa
#                 center_weights = [1 + (1 - abs(move - COLUMNS//2)/(COLUMNS//2)) for move, _ in top_moves]
#                 total_weight = sum(center_weights)
#                 weights = [w/total_weight for w in center_weights]
#                 moves = [move for move, _ in top_moves]
#                 return random.choices(moves, weights=weights, k=1)[0]
                
#             weights = [score / total_score if score > 0 else 0.01 for _, score in top_moves]
#             total_weight = sum(weights)
#             norm_weights = [w/total_weight for w in weights]
#             moves = [move for move, _ in top_moves]
                
#             return random.choices(moves, weights=norm_weights, k=1)[0]
            
#         return top_moves[0][0] 
        
#     # Ưu tiên cột giữa
#     if COLUMNS // 2 in valid_moves:
#         return COLUMNS // 2

#     return random.choice(valid_moves)

# def is_endgame(board):
#     """Xác định xem đang ở giai đoạn endgame hay không
#     Endgame: Khi có >70% bàn cờ đã lấp đầy hoặc có nhiều 3 quân liên tiếp
#     """
#     # Đếm số ô đã được lấp đầy
#     filled_count = np.count_nonzero(board != EMPTY) if isinstance(board, np.ndarray) else sum(1 for r in board for c in r if c != EMPTY)
#     total_cells = ROWS * COLUMNS
    
#     # Nếu đã lấp đầy >70% là endgame
#     if filled_count >= 0.7 * total_cells:
#         return True
    
#     # Kiểm tra 3 quân liên tiếp
#     for player_piece in [PLAYER, AI]:
#         # Kiểm tra theo hàng ngang
#         for r in range(ROWS):
#             for c in range(COLUMNS-3):
#                 window = [board[r][c+i] for i in range(4)]
#                 if window.count(player_piece) == 3 and window.count(EMPTY) == 1:
#                     return True
        
#         # Kiểm tra theo hàng dọc
#         for c in range(COLUMNS):
#             for r in range(ROWS-3):
#                 window = [board[r+i][c] for i in range(4)]
#                 if window.count(player_piece) == 3 and window.count(EMPTY) == 1:
#                     return True
        
#         # Kiểm tra theo đường chéo /
#         for r in range(ROWS-3):
#             for c in range(COLUMNS-3):
#                 window = [board[r+i][c+i] for i in range(4)]
#                 if window.count(player_piece) == 3 and window.count(EMPTY) == 1:
#                     return True
        
#         # Kiểm tra theo đường chéo \
#         for r in range(3, ROWS):
#             for c in range(COLUMNS-3):
#                 window = [board[r-i][c+i] for i in range(4)]
#                 if window.count(player_piece) == 3 and window.count(EMPTY) == 1:
#                     return True
    
#     return False

# def rollout(board, player):
#     """Mô phỏng ván chơi từ trạng thái hiện tại với 2 chế độ khác nhau"""
#     # Giới hạn độ sâu để tăng tốc
#     max_rollout_depth = 15
    
#     # Quyết định sử dụng rollout nào
#     use_heavy = is_endgame(board)
    
#     current_board = board.copy()
#     opponent = PLAYER if player == AI else AI
#     current_player = player
#     depth = 0
    
#     while depth < max_rollout_depth:
#         if winning_move(current_board, player)[0]:
#             return 1.0  # Thắng
#         elif winning_move(current_board, opponent)[0]:
#             return 0.0  # Thua
        
#         valid_moves = get_valid_locations(current_board)
#         if not valid_moves:
#             return 0.5  # Hòa
        
#         # Lựa chọn policy tùy theo giai đoạn
#         if use_heavy:
#             move = heavy_rollout_policy(current_board, current_player)
#         else:
#             move = light_rollout_policy(current_board, current_player)
            
#         row = get_next_open_row(current_board, move)
#         if row is not None:
#             drop_piece(current_board, row, move, current_player)
            
#             current_player = PLAYER if current_player == AI else AI
#             depth += 1
#         else:
#             break
    
#     # Nếu đạt giới hạn độ sâu mà không có kết quả, đánh giá trạng thái
#     if depth >= max_rollout_depth:
#         # Đánh giá nhanh
#         player_score = evaluate_board(current_board, player)
#         opponent_score = evaluate_board(current_board, opponent)
        
#         total_score = abs(player_score) + abs(opponent_score)
#         if total_score == 0:
#             return 0.5
        
#         normalized_score = (player_score + total_score) / (2 * total_score)
#         return min(max(normalized_score, 0.0), 1.0)
    
#     # Nếu trận đấu kết thúc
#     if winning_move(current_board, player)[0]:
#         return 1.0
#     elif winning_move(current_board, opponent)[0]:
#         return 0.0
#     else:
#         return 0.5  # Hòa

# class MCTSNode:
#     def __init__(self, board, parent=None, move=None, player=None):
#         self.board = board.copy()
#         self.parent = parent
#         self.move = move  
#         self.player = player  
#         self.children = []  
#         self.wins = 0 
#         self.visits = 0 
#         self.untried_moves = get_valid_locations(board)  
#         self.terminal = False  
#         self.total_value = 0.0
        
#         if winning_move(board, AI)[0] or winning_move(board, PLAYER)[0] or len(self.untried_moves) == 0:
#             self.terminal = True
    
#     def uct_select_child(self):
#         unvisited = [child for child in self.children if child.visits == 0]
#         if unvisited:
#             return random.choice(unvisited)
        
#         log_total_visits = math.log(self.visits)
        
#         return max(self.children, key=lambda c: c.get_uct_value(log_total_visits))
            
#     def get_uct_value(self, log_parent_visits):
#         if self.visits == 0:
#             return float('inf')
        
#         avg_value = self.total_value / self.visits
        
#         exploration = C_PARAM * math.sqrt(log_parent_visits / self.visits)
        
#         center_bias = 0
#         if self.move is not None:
#             center_bias = (1.0 - abs(self.move - COLUMNS // 2) / (COLUMNS // 2)) * 0.1
        
#         return avg_value + exploration + center_bias
    
#     def expand(self):
#         if not self.untried_moves or self.terminal:
#             return None
        
#         # Ưu tiên các nước đi ở giữa bàn cờ
#         center_col = COLUMNS // 2
#         weighted_moves = []
#         for move in self.untried_moves:
#             weight = 1.0 + (1.0 - abs(move - center_col) / center_col) * 2.0
#             weighted_moves.append((move, weight))
        
#         moves, weights = zip(*weighted_moves)
#         total_weight = sum(weights)
#         normalized_weights = [w / total_weight for w in weights]

#         # Random bias về hướng cột middle.
#         move = np.random.choice(moves, p=normalized_weights)
        
#         self.untried_moves.remove(move)
        
#         child_board = self.board.copy()
#         row = get_next_open_row(child_board, move)
#         if row is not None:
#             drop_piece(child_board, row, move, self.player)
            
#             next_player = PLAYER if self.player == AI else AI
#             child_node = MCTSNode(child_board, self, move, next_player)
#             self.children.append(child_node)
            
#             return child_node
        
#         return None
    
#     def update(self, result):
#         self.visits += 1
#         self.total_value += result
    
#     def is_fully_expanded(self):
#         return len(self.untried_moves) == 0
    
#     def is_terminal(self):
#         return self.terminal
    
#     def get_best_move(self):
#         if not self.children:
#             return None
        
#         visited_children = [child for child in self.children if child.visits > 0]
#         if not visited_children:
#             return None
            
#         best_child = max(visited_children, key=lambda c: c.visits)
        
#         high_win_rate_children = [child for child in visited_children 
#                                  if child.visits >= max(100, best_child.visits * 0.3) and 
#                                  child.total_value / child.visits >= 0.9]
        
#         if high_win_rate_children:
#             return max(high_win_rate_children, 
#                       key=lambda c: (c.total_value / c.visits) + 0.0001 * c.visits).move
        
#         return int(best_child.move)

# def mcts_search(board, player, simulations=MIN_SIMULATIONS, time_limit=TIME_LIMIT):
#     """Tìm kiếm nước đi tốt nhất bằng MCTS"""
#     # Reset cache cho mỗi lần tìm kiếm mới
#     global board_eval_cache
#     board_eval_cache = {}
    
#     if not isinstance(board, np.ndarray):
#         board = np.array(board)
    
#     # Thực hiện kiểm tra thắng/thua nhanh trước khi bắt đầu MCTS
#     for col in range(COLUMNS):
#         if is_valid_location(board, col):
#             row = get_next_open_row(board, col)
#             if row is not None:
#                 board_copy = board.copy()
#                 drop_piece(board_copy, row, col, player)
#                 if winning_move(board_copy, player)[0]:
#                     return col
    
#     opponent = PLAYER if player == AI else AI
#     for col in range(COLUMNS):
#         if is_valid_location(board, col):
#             row = get_next_open_row(board, col)
#             if row is not None:
#                 board_copy = board.copy()
#                 drop_piece(board_copy, row, col, opponent)
#                 if winning_move(board_copy, opponent)[0]:
#                     return col
                    
#     two_way_move, _ = detect_two_way_win(board, player)
#     if two_way_move is not None:
#         return two_way_move

#     opponent_two_way, _ = detect_two_way_win(board, opponent)
#     if opponent_two_way is not None:
#         return opponent_two_way

#     forced_win_move = find_forced_win(board, player, 3)  # Giảm độ sâu từ 4 xuống 3
#     if forced_win_move is not None:
#         return forced_win_move
    
#     # Ưu tiên cột giữa nếu là bàn cờ trống
#     is_empty_board = True
#     for r in range(ROWS):
#         for c in range(COLUMNS):
#             if board[r, c] != EMPTY:
#                 is_empty_board = False
#                 break
#         if not is_empty_board:
#             break
            
#     if is_empty_board:
#         return COLUMNS // 2
    
#     # Ưu tiên cột giữa nếu còn trống
#     if board[ROWS-1, COLUMNS//2] == EMPTY:
#         return COLUMNS // 2
    
#     # Bắt đầu MCTS
#     root = MCTSNode(board, player=player)
    
#     if len(root.untried_moves) == 1:
#         return root.untried_moves[0]

#     start_time = time.time()
#     sim_count = 0
    
#     max_time = time_limit * 0.95  # Sử dụng 95% thời gian
    
#     # Xác định xem có đang ở endgame không - để ghi log
#     endgame = is_endgame(board)
#     rollout_type = "Heavy rollout (endgame)" if endgame else "Light rollout (midgame)"
    
#     while ((time.time() - start_time) < max_time) and (sim_count < simulations):
#         node = root
        
#         # Selection
#         while node.is_fully_expanded() and not node.is_terminal():
#             node = node.uct_select_child()
        
#         # Expansion
#         if not node.is_terminal():
#             new_node = node.expand()
#             if new_node:
#                 node = new_node

#         # Simulation
#         result = rollout(node.board, node.player)
  
#         # Backpropagation
#         while node:
#             if node.player == player:
#                 node.update(1 - result)
#             else:
#                 node.update(result)
#             node = node.parent
        
#         sim_count += 1
        
#         # Chỉ kiểm tra mỗi 200 mô phỏng để giảm chi phí
#         if sim_count % 200 == 0:
#             best_child = max(root.children, key=lambda c: c.visits) if root.children else None
#             if best_child and best_child.visits > sim_count * 0.4:  # Nếu một nước chiếm >40% số lần thăm
#                 win_rate = best_child.total_value / best_child.visits
#                 if win_rate > 0.95 and sim_count > simulations // 4:
#                     break
                
#                 if sim_count > simulations // 2 and win_rate > 0.8:
#                     second_best = sorted(root.children, key=lambda c: c.visits)[-2] if len(root.children) >= 2 else None
#                     if second_best and best_child.visits > second_best.visits * 1.5:
#                         break
    
#     elapsed_time = time.time() - start_time
#     print(f"MCTS ({rollout_type}): {sim_count} mô phỏng trong {elapsed_time:.3f}s ({elapsed_time/time_limit*100:.1f}% thời gian)")

#     if root.children:
#         print(f"Thống kê mô phỏng theo cột:")
#         total_visits = sum(child.visits for child in root.children)
        
#         for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
#             if child.visits > 0:
#                 win_rate = child.total_value / child.visits
#                 win_percent = win_rate * 100
#                 visit_percent = (child.visits / total_visits) * 100 if total_visits > 0 else 0
#                 print(f"  Cột {child.move}: {child.visits} lần thăm ({visit_percent:.1f}% tổng số), tỷ lệ thắng: {win_percent:.1f}%")

#     # Đợi thời gian còn lại
#     remaining_time = max_time - elapsed_time
#     if remaining_time > 0:
#         print(f"Chờ thêm {remaining_time:.3f}s để đảm bảo thời gian đồng bộ")
#         time.sleep(max(0, remaining_time))
    
#     return int(root.get_best_move())

# def main():
#     matrix = [
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 2, 0, 0, 0],
#         [0, 0, 0, 2, 1, 0, 0],
#         [0, 0, 2, 1, 1, 0, 0]
#     ]

#     move = mcts_search(matrix, PLAYER)
#     print(f"Nước đi tốt nhất: {move}")

# if __name__ == "__main__":
#     main()