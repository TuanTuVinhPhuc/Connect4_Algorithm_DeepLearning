import numpy as np
import math
import random
import time
from functools import lru_cache
from collections import OrderedDict
from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
from board import is_valid_location, get_next_open_row, get_valid_locations, drop_piece, winning_move

C_PARAM = 0.9
MIN_SIMULATIONS = 40000
TIME_LIMIT = 10

#LRU cache
board_eval_cache = OrderedDict()
MAX_CACHE_SIZE = 100000

# board:
    # 0 1 2 3 4 5 6
    # 1 1 2 3 4 5 6 
    # 2 1 2 3 4 5 6
    # 3 1 2 3 4 5 6
    # 4 1 2 3 4 5 6 
    # 5 1 2 3 4 5 6

class MCTSNode:
    def __init__(self, board, parent=None, move=None, player=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  
        self.player = player  
        self.children = []  
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
            center_bias = (1.0 - abs(self.move - COLUMNS // 2) / (COLUMNS // 2)) * 0.3
        
        return avg_value + exploration + center_bias

    def expand(self):
        if not self.untried_moves or self.terminal:
            return None

        center_col = COLUMNS // 2
        weighted_moves = []
        claimeven_factor = 1.5
        baseinverse_factor = 2.0
        oddthreat_factor = 1.2

        # Cache get_next_open_row
        row_cache = {move: get_next_open_row(self.board, move) for move in self.untried_moves}
        # Vector hóa piece_count
        piece_counts = np.count_nonzero(self.board != EMPTY, axis=0)

        for move in self.untried_moves:
            row = row_cache[move]
            if row is None:
                continue
            center_weight = 1.0 + (1.0 - abs(move - center_col) / center_col) * 3.0
            if move == center_col:
                center_weight *= 1.5 # Thêm chút bias cho center.
            claimeven_weight = claimeven_factor if row in [1, 3, 5] else 1.0
            piece_count = piece_counts[move]
            baseinverse_weight = 1.0
            if piece_count <= 2 and row in [1, 3, 5]:
                baseinverse_weight = baseinverse_factor if row == 5 else baseinverse_factor * 0.75
            oddthreat_weight = oddthreat_factor if row in [0, 2, 4] else 1.0
            weight = center_weight * claimeven_weight * baseinverse_weight * oddthreat_weight
            weighted_moves.append((move, weight))

        if not weighted_moves:
            return None

        moves, weights = zip(*weighted_moves)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        move = np.random.choice(moves, p=normalized_weights)

        self.untried_moves.remove(move)
        child_board = self.board.copy()
        row = row_cache[move]
        if row is not None:
            child_board[row, move] = self.player
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
    
# evaluate
@lru_cache(maxsize=1024)
def evaluate_window(window_tuple, piece, position):
    window = list(window_tuple)
    opp_piece = PLAYER if piece == AI else AI
    
    piece_count = window.count(piece)
    empty_count = window.count(EMPTY)
    opp_count = window.count(opp_piece)

    score = 0
    SCORES = {
    "win": 10000,
    "open_three": 8,
    "blocked_three": 5,
    "connected_two": 2,
    "threat_block_open": -12,
    "threat_block": -9,
    "opp_two": -2
    }

    start_row, start_col, direction = position
    center_factor = 1.0

    if direction == 0:
        center_factor = 1.0 + 0.2 * (1.0 - abs(start_col + 1.5 - COLUMNS/2)/(COLUMNS/2))
        
    elif direction == 1:
        center_factor = 1.0 + 0.2 * (1.0 - abs(start_col - COLUMNS/2)/(COLUMNS/2))
        
    else:
        row_center = 1.0 - abs(start_row + 1.5 - ROWS/2)/(ROWS/2)
        col_center = 1.0 - abs(start_col + 1.5 - COLUMNS/2)/(COLUMNS/2)
        center_factor = 1.0 + 0.15 * (row_center + col_center)/2
    
    if piece_count == 4:
        return SCORES["win"] * center_factor
    elif piece_count == 3 and empty_count == 1:
        score += SCORES["blocked_three"] * center_factor
    elif piece_count == 2 and empty_count == 2:
        score += SCORES["connected_two"] * center_factor
    
    if opp_count == 3 and empty_count == 1:
        score += SCORES["threat_block"] * center_factor
    elif opp_count == 2 and empty_count == 2:
        score += SCORES["opp_two"] * center_factor
    
    return score * center_factor

# Bộ nhớ cache cho evaluate_board
board_eval_cache = {}
def evaluate_board(board, piece):
    if not isinstance(board, (list, np.ndarray)) or len(board) != ROWS or len(board[0]) != COLUMNS:
        raise ValueError("Invalid board format")
    
    board_key = tuple(map(tuple, board)) if isinstance(board, list) else hash(board.tobytes())
    cache_key = (board_key, piece)
    
    if cache_key in board_eval_cache:
        value = board_eval_cache.pop(cache_key)
        board_eval_cache[cache_key] = value
        return board_eval_cache[cache_key]
    
    score = 0
    SCORES = {
        "win": 10000,
        "open_three": 12,
        "blocked_three": 8,
        "connected_two": 2,
        "threat_block_open": -12,
        "threat_block": -9,
        "opp_two": -2
    }

    center_array = [board[r][COLUMNS//2] for r in range(ROWS)]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    # Kiểm tra hàng ngang
    for r in range(ROWS):
        for c in range(COLUMNS-3):
            window = [board[r][c+i] for i in range(4)]
            position = (r, c, 0)  # 0 = ngang
            base_score = evaluate_window(tuple(window), piece, position)
            if base_score == SCORES["blocked_three"] and c > 0 and c + 4 < COLUMNS:
                empty_pos = window.index(EMPTY)
                if (empty_pos == 0 and board[r][c + 4] == EMPTY) or \
                   (empty_pos == 3 and board[r][c - 1] == EMPTY):
                    base_score = SCORES["open_three"]
            if base_score == SCORES["threat_block"] and c > 0 and c + 4 < COLUMNS:
                empty_pos = window.index(EMPTY)
                if (empty_pos == 0 and board[r][c + 4] == EMPTY) or \
                   (empty_pos == 3 and board[r][c - 1] == EMPTY):
                    base_score = SCORES["threat_block_open"]
            score += base_score
    
    # Kiểm tra hàng dọc
    for c in range(COLUMNS):
        for r in range(ROWS-3):
            window = [board[r+i][c] for i in range(4)]
            position = (r, c, 1)  # 1 = dọc
            score += evaluate_window(tuple(window), piece, position)
    
    # Kiểm tra đường chéo \
    for r in range(ROWS-3):
        for c in range(COLUMNS-3):
            window = [board[r+i][c+i] for i in range(4)]
            position = (r, c, 2)  # 2 = chéo xuống
            base_score = evaluate_window(tuple(window), piece, position)
            if base_score == SCORES["blocked_three"] and r > 0 and c > 0 and r + 4 < ROWS and c + 4 < COLUMNS:
                empty_pos = window.index(EMPTY)
                if (empty_pos == 0 and board[r + 4][c + 4] == EMPTY) or \
                   (empty_pos == 3 and board[r - 1][c - 1] == EMPTY):
                    base_score = SCORES["open_three"]
            if base_score == SCORES["threat_block"] and r > 0 and c > 0 and r + 4 < ROWS and c + 4 < COLUMNS:
                empty_pos = window.index(EMPTY)
                if (empty_pos == 0 and board[r + 4][c + 4] == EMPTY) or \
                   (empty_pos == 3 and board[r - 1][c - 1] == EMPTY):
                    base_score = SCORES["threat_block_open"]
            score += base_score
    
    # Kiểm tra đường chéo /
    for r in range(3, ROWS):
        for c in range(COLUMNS-3):
            window = [board[r-i][c+i] for i in range(4)]
            position = (r, c, 3)  # 3 = chéo lên
            base_score = evaluate_window(tuple(window), piece, position)
            if base_score == SCORES["blocked_three"] and r < ROWS - 1 and c > 0 and r - 4 >= 0 and c + 4 < COLUMNS:
                empty_pos = window.index(EMPTY)
                if (empty_pos == 0 and board[r - 4][c + 4] == EMPTY) or \
                   (empty_pos == 3 and board[r + 1][c - 1] == EMPTY):
                    base_score = SCORES["open_three"]
            if base_score == SCORES["threat_block"] and r < ROWS - 1 and c > 0 and r - 4 >= 0 and c + 4 < COLUMNS:
                empty_pos = window.index(EMPTY)
                if (empty_pos == 0 and board[r - 4][c + 4] == EMPTY) or \
                   (empty_pos == 3 and board[r + 1][c - 1] == EMPTY):
                    base_score = SCORES["threat_block_open"]
            score += base_score
    
    board_eval_cache[cache_key] = score
    if len(board_eval_cache) > MAX_CACHE_SIZE * 2: 
        board_eval_cache.popitem()
        
    return score

two_way_win_cache = {}
def detect_two_way_win(board, player):
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
    if len(two_way_win_cache) > 20000:
        two_way_win_cache.popitem(last=False)
    return result

# Cache cho find_forced_win
forced_win_cache = {}
def find_forced_win_move(board, player, depth, alpha=-float('inf'), beta=float('inf')):
    if depth <= 0:
        return None, 0
    
    board_key = tuple(map(tuple, board))
    cache_key = (board_key, player, depth)
    
    if cache_key in forced_win_cache:
        return forced_win_cache[cache_key]
        
    valid_moves = get_valid_locations(board)
    opponent = PLAYER if player == AI else AI
    
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
 
        board[row][move] = player
        
        if winning_move(board, player)[0]:
            board[row][move] = EMPTY
            result = (move, 1000)  
            forced_win_cache[cache_key] = result
            return result

        board[row][move] = EMPTY

    two_way_move, threats = detect_two_way_win(board, player)
    if two_way_move is not None:
        result = (two_way_move, 900) 
        forced_win_cache[cache_key] = result
        return result

    best_move = None
    best_value = -float('inf')

    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
 
        board[row][move] = player

        min_value = float('inf')
 
        opponent_valid_moves = get_valid_locations(board)

        opponent_moves_scored = []
        for opp_move in opponent_valid_moves:
            opp_row = get_next_open_row(board, opp_move)
            if opp_row is None:
                continue

            board[opp_row][opp_move] = opponent
            if winning_move(board, opponent)[0]:
                score = -1000  
            else:
                center_col = COLUMNS // 2
                score = -(1.0 - abs(opp_move - center_col) / center_col) * 10
            board[opp_row][opp_move] = EMPTY
            
            opponent_moves_scored.append((opp_move, score))

        opponent_moves_scored.sort(key=lambda x: x[1])
        opponent_moves = [move for move, _ in opponent_moves_scored]
        
        for opponent_move in opponent_moves:
            opponent_row = get_next_open_row(board, opponent_move)
            if opponent_row is None:
                continue
                
            board[opponent_row][opponent_move] = opponent
            
            if winning_move(board, opponent)[0]:
                value = -800  
            else:
                next_move, value = find_forced_win_move(board, player, depth-1, alpha, beta)
                if next_move is None:
                    value = -100  # Không tìm thấy forced win
   
            board[opponent_row][opponent_move] = EMPTY
            
            min_value = min(min_value, value)
            
            # Alpha-beta pruning
            beta = min(beta, min_value)
            if alpha >= beta:
                break  # Beta cutoff
        
        board[row][move] = EMPTY
        
        if min_value > 0 and min_value > best_value:
            best_value = min_value
            best_move = move
            
        alpha = max(alpha, best_value)
        if alpha >= beta:
            break  # Alpha cutoff
    
    result = (best_move, best_value if best_move is not None else 0)
    forced_win_cache[cache_key] = result
    return result

def find_forced_win(board, player, depth):
    move, _ = find_forced_win_move(board, player, depth)
    return move

def check_open_three_threat(board, opponent):
    valid_moves = get_valid_locations(board)
    if not valid_moves:
        return None, None
    
    board = np.array(board)
    
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue

        board[row, move] = opponent
        threat_found = False
        threat_row = None
  
        for r in range(ROWS):
            windows = np.lib.stride_tricks.sliding_window_view(board[r], 4)
            for c in range(len(windows)):
                window = windows[c]
                if (np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == EMPTY) == 1):
                    threat_found = True
                    threat_row = r
                    break
            if threat_found:
                break
        
        # Kiểm tra hàng dọc
        if not threat_found:
            for c in range(COLUMNS):
                windows = np.lib.stride_tricks.sliding_window_view(board[:, c], 4)
                for r in range(len(windows)):
                    window = windows[r]
                    if (np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == EMPTY) == 1 and row in [1, 3, 5]):
                        threat_found = True
                        empty_pos = np.where(window == EMPTY)[0][0]
                        threat_row = r + empty_pos
                        break
                if threat_found:
                    break
 
        if not threat_found:
            for r in range(ROWS-3):
                for c in range(COLUMNS-3):
                    window = board[r:r+4, c:c+4].diagonal()
                    if np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == EMPTY) == 1:
                        threat_found = True
                        empty_pos = np.where(window == EMPTY)[0][0]
                        threat_row = r + empty_pos
                        break
                if threat_found:
                    break
 
        if not threat_found:
            for r in range(3, ROWS):
                for c in range(COLUMNS-3):
                    window = np.fliplr(board[r-3:r+1, c:c+4]).diagonal()
                    if np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == EMPTY) == 1:
                        threat_found = True
                        empty_pos = np.where(window == EMPTY)[0][0]
                        threat_row = r - empty_pos
                        break
                if threat_found:
                    break
        
        # Undo
        board[row, move] = EMPTY
        
        if threat_found:
            return move, threat_row
    
    return None, None

def rollout_policy(board, player):
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
    
    open_three_way, _ = check_open_three_threat(board, opponent)
    if open_three_way is not None:
        return open_three_way
    
    # Kiểm tra forced win
    pieces_count = sum((row == PLAYER).sum() + (row == AI).sum() for row in board)
    if (pieces_count > 15): 
       forced_win = find_forced_win(board, player, 4)
       if forced_win is not None:
            return forced_win
    
    # Tính điểm cho các nước đi
    move_scores = []
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        board[row][move] = player
        score = evaluate_board(board, player)
        center_preference = (1.0 - abs(move - COLUMNS // 2) / (COLUMNS // 2)) * 3.5
        score += center_preference
        
        # Kiểm tra chuỗi 3 quân cục bộ
        for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for offset in [-3, -2, -1, 0]:
                r, c = row + offset * direction[0], move + offset * direction[1]
                if 0 <= r <= ROWS-4 and 0 <= c <= COLUMNS-4:
                    window = [board[r + i*direction[0]][c + i*direction[1]] for i in range(4)]
                    if window.count(player) == 3 and window.count(EMPTY) == 1:
                        score += 3
                    elif window.count(opponent) == 3 and window.count(EMPTY) == 1:
                        score -= 6
        
        board[row][move] = EMPTY
        move_scores.append((move, score))
            
    if move_scores:
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_moves = move_scores[:min(3, len(move_scores))]
        
        if len(top_moves) > 1:
            if top_moves[0][1] > 0 and top_moves[0][1] >= top_moves[1][1] + 5:
                return top_moves[0][0]
            
            total_score = sum(score for _, score in top_moves)
            if total_score != 0:  
                weights = [score / total_score for _, score in top_moves]
                moves = [move for move, _ in top_moves]
                 
                return random.choices(moves, weights=weights, k=1)[0]
            
        return top_moves[0][0] 
    
    if COLUMNS // 2 in valid_moves:
        return COLUMNS // 2

    return random.choice(valid_moves)
    
def rollout(board, player):
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
    
    score = evaluate_board(current_board, player)
    if score == 0:
        return 0.5
    normalized_score = 0.5 + score / (2 * (abs(score) + 1))
    return min(max(normalized_score, 0.0), 1.0)   

def mcts_search(board, player, simulations=MIN_SIMULATIONS, time_limit=TIME_LIMIT):
    global board_eval_cache, two_way_win_cache, forced_win_cache
    board_eval_cache = {}
    two_way_win_cache = {}
    forced_win_cache = {}
    
    if not isinstance(board, np.ndarray):
        board = np.array(board)
    
    opponent = PLAYER if player == AI else AI

    # Thực hiện kiểm tra thắng/thua nhanh trước khi bắt đầu MCTS
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is not None:
                board[row][col] = player
                if winning_move(board, player)[0]:
                    board[row][col] = EMPTY
                    return col
                board[row][col] = EMPTY

                if winning_move(board, opponent)[0]:
                    board[row][col] = EMPTY
                    return col
                board[row][col] = EMPTY
                    
    two_way_move, _ = detect_two_way_win(board.tolist(), player)
    if two_way_move is not None:
        print("sủ dụng chiến thuật 2 nước thắng")
        return two_way_move

    opponent_two_way, _ = detect_two_way_win(board.tolist(), opponent)
    if opponent_two_way is not None:
        print("sủ dụng chiến thuật chặn 2 nước thắng")
        return opponent_two_way
    
    open_three_way, threat_row = check_open_three_threat(board, opponent)
    if open_three_way is not None and threat_row > 2:
        print("sử dụng check_open_three_threat")
        print("threat: ", threat_row)
        return open_three_way
    
    pieces_count = sum((row == PLAYER).sum() + (row == AI).sum() for row in board)
    if (pieces_count > 10):
        forced_win_move = find_forced_win(board.tolist(), player, 4)
        if forced_win_move is not None:
            print("sủ dụng chiến thuật chắc chắn thắng")
            return forced_win_move
    
    # Ưu tiên cột giữa nếu là bàn cờ trống
    is_empty_board = np.all(board == EMPTY)
    if is_empty_board or board[ROWS-1, COLUMNS//2] == EMPTY:
        return COLUMNS // 2
    
    # Bắt đầu MCTS
    root = MCTSNode(board, player=player)
    
    if len(root.untried_moves) == 1:
        return root.untried_moves[0]

    start_time = time.time()
    sim_count = 0
    
    max_time = time_limit * 0.95
    
    pieces_count = np.count_nonzero(board != 0)
    time_factor = min(1.0, pieces_count / (ROWS * COLUMNS * 0.7))
    
    actual_max_time = max_time * (0.95 + 0.05 * time_factor)
    
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
        
        if sim_count % 100 == 0 and sim_count > simulations // 4:  # Kiểm tra định kỳ

            if root.children:
                sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
                best_child = sorted_children[0]
                second_best = sorted_children[1] if len(sorted_children) > 1 else None
                if best_child.visits > 0:
                    win_rate = best_child.total_value / best_child.visits
                    if (win_rate > 0.9 and sim_count > simulations * 0.75) or \
                       (second_best and best_child.visits > second_best.visits * 2 and win_rate > 0.85):
                        print(f"Dừng sớm: Cột {best_child.move}, win_rate={win_rate:.2f}, visits={best_child.visits}")
                        break
    
    elapsed_time = time.time() - start_time
    print(f"MCTS: {sim_count} mô phỏng trong {elapsed_time:.3f}s ({elapsed_time/time_limit*100:.1f}% thời gian)")
    if root.children:
        total_visits = sum(child.visits for child in root.children)
        print(f"Thống kê mô phỏng theo cột:")
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
            if child.visits > 0:
                win_rate = child.total_value / child.visits
                win_percent = win_rate * 100
                visit_percent = (child.visits / total_visits) * 100 if total_visits > 0 else 0
                print(f"  Cột {child.move}: {child.visits} lần thăm ({visit_percent:.1f}% tổng số), tỷ lệ thắng: {win_percent:.1f}%")
    
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

    print(mcts_search(matrix, PLAYER))
if __name__ == "__main__":
    main()
