import math
from functools import lru_cache
import time
import numpy as np
from functools import lru_cache
from collections import OrderedDict
from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
from board import get_next_open_row, winning_move, get_valid_locations, is_terminal_node, is_valid_location

board_eval_cache = OrderedDict()
MAX_CACHE_SIZE = 100000

def checking_winning(board, piece):
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is None:
                continue
            # Thực hiện nước đi trực tiếp thay vì copy
            board[row][col] = piece
            win = winning_move(board, piece)[0]
            # Hoàn tác
            board[row][col] = EMPTY
            
            if win:
                return col
    return None

def evaluate_window(window, piece, position):
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

board_eval_cache = {}
def score_position(board, piece):
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
    
@lru_cache(maxsize=10000)
def cached_score_position(board_tuple, piece):
    # Chuyển tuple thành array để xử lý
    board = np.array(board_tuple).reshape((ROWS, COLUMNS))
    return score_position(board, piece)

def detect_two_way_win(board, piece):
    valid_locations = get_valid_locations(board)

    for move in valid_locations:
       row = get_next_open_row(board, move)
       if row is None:
           continue
           
       board_copy = board.copy()
       board_copy[row][move] = piece
       
       win_threats = 0
       for next_move in valid_locations:
           if next_move == move:
               continue
           
           next_row = get_next_open_row(board_copy, next_move)
           if next_row is None:
               continue
           
           next_board = board_copy.copy()
           next_board[next_row][next_move] = piece
           
           if winning_move(next_board, piece)[0]:
               win_threats += 1
               if win_threats >= 2:
                   return move
    return None

def is_symmetric(board):
    for r in range(ROWS):
        for c in range(COLUMNS // 2):
            if board[r][c] != board[r][COLUMNS-1-c]:
                return False
    return True

def get_symmetry_factor(board):
    if is_symmetric(board):
        piece_count = np.count_nonzero(board)
        return max(0.2, min(0.8, 0.2 + piece_count * 0.02))
    else:
        return 1.0

def filter_symmetric_moves(board, valid_locations):
    if np.count_nonzero(board) <= 2:
        return [col for col in valid_locations if col >= COLUMNS // 2]
    
    symmetric = True
    for r in range(ROWS):
        for c in range(COLUMNS // 2):
            if board[r][c] != board[r][COLUMNS-1-c]:
                symmetric = False
                break
        if not symmetric:
            break
    
    if symmetric:
        filtered = []
        seen = set()
        
        for col in valid_locations:
            symmetric_col = COLUMNS - 1 - col
            if col >= COLUMNS // 2 or symmetric_col in seen:
                filtered.append(col)
            seen.add(col)
        
        return filtered
    
    return valid_locations

def order_moves(board, valid_locations, current_player, history=None):
    if history is None:
        history = {}
    
    filtered_locations = filter_symmetric_moves(board, valid_locations)

    move_scores = []
    
    for col in filtered_locations:
        score = 0

        # Lấy điểm từ history nếu có
        if (col, current_player) in history:
            score += history[(col, current_player)]
            
        row = get_next_open_row(board, col)
        if row is not None:
            # Tạo bàn cờ tạm để đánh giá
            temp_board = board.copy()
            temp_board[row][col] = current_player
            eval_score = score_position(temp_board, current_player)
            score += eval_score

            # Ưu tiên hàng chẵn (1, 3, 5)
            if row % 2 == 1:  
                Claimeven_score = 15  # Giá trị ưu tiên cho hàng chẵn
                Baseinverse_score = row * 1.5  # Ưu tiên hàng thấp hơn 
                score = Claimeven_score + Baseinverse_score
                
        move_scores.append((col, score))
        
    # Sắp xếp theo điểm giảm dần
    move_scores.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in move_scores]

def minimax_with_prunning(board, depth, alpha, beta, max_player, min_player, maximizing_player=True, history=None, time_limit=3.0, start_time=None):
    if history is None:
        history = {}

    if time_limit and start_time and time.time() - start_time > time_limit:
        return None, 0
    
    valid_locations = get_valid_locations(board)
    if not valid_locations:
        return None, 0
    
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, max_player)[0]:  
                return (None, 1e5 + depth)
            elif winning_move(board, min_player)[0]:  
                return (None, -1e5 - depth)
            else:  # Hết nước đi (hòa)
                return (None, 0)
        else:  # Đến độ sâu 0
            sym_factor = get_symmetry_factor(board)
            return (None, cached_score_position(tuple(map(tuple, board)), max_player) * sym_factor)
        
    # Sắp xếp các nước đi theo thứ tự ưu tiên (Move Ordering)
    ordered_locations = order_moves(board, valid_locations, max_player, history)
    
    if maximizing_player: 
        value = -math.inf
        column = ordered_locations[0] if ordered_locations else None
        
        for col in ordered_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
        
            # Thực hiện nước đi
            board[row][col] = max_player
            
            # Đánh giá
            new_score = minimax_with_prunning(board, depth-1, alpha, beta, max_player, min_player, False, history, time_limit, start_time)[1]
            
            # Hoàn tác nước đi
            board[row][col] = EMPTY
            
            # Cập nhật history cho Move Ordering
            history[(col, maximizing_player)] = new_score
            
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Alpha-Beta pruning
                
        return column, value
    else:  
        value = math.inf
        column = ordered_locations[0] if ordered_locations else None
        
        for col in ordered_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
        
            # Thực hiện nước đi
            board[row][col] = min_player
            
            # Đánh giá
            new_score = minimax_with_prunning(board, depth-1, alpha, beta, max_player, min_player, True, history, time_limit, start_time)[1]
            
            # Hoàn tác nước đi
            board[row][col] = EMPTY
            
            # Cập nhật history cho Move Ordering
            history[(col, maximizing_player)] = -new_score
            
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha-Beta pruning
                
        return column, value

def iterative_deepening_minimax(board, max_depth, alpha, beta, max_player , min_player, maximizing_player=True, time_limit=3.0):
    start_time = time.time()
    best_move = None
    best_score = -math.inf if maximizing_player else math.inf
    history = {}  # Lưu lại thông tin nước đi từ các độ sâu trước
    
    # Bắt đầu từ độ sâu 1, tăng dần lên max_depth
    for current_depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit * 0.9:
            # Nếu đã sử dụng 90% thời gian, dừng lại
            break
            
        move, score = minimax_with_prunning(
            board, current_depth, alpha, beta, max_player , min_player, maximizing_player, 
            history, time_limit * 0.9, start_time
        )
        
        if move is None:
            # Hết thời gian trong quá trình tìm kiếm
            break
            
        # Cập nhật best_move nếu tìm thấy nước tốt hơn
        if maximizing_player and score > best_score:
            best_move = move
            best_score = score
        elif not maximizing_player and score < best_score:
            best_move = move
            best_score = score
            
        # Nếu đã tìm thấy nước thắng hoặc thua chắc chắn, không cần tìm sâu hơn
        if abs(score) > 1e4:
            break
            
    return best_move, best_score

def minimax(board, depth, alpha, beta, max_player, min_player, maximizing_player, time_limit=3.0):
    if not isinstance(board, np.ndarray):
        board = np.array(board)

    total_pieces = np.count_nonzero(board != 0)

    if total_pieces >= 6:
        # Nếu có nước thắng ngay
        win_col = checking_winning(board, max_player)
        if win_col is not None:
            return win_col, 1e5 

        # Nếu cần chặn đối thủ
        block_col = checking_winning(board, min_player)
        if block_col is not None:
            return block_col, 9e4 

    trap_col_min = detect_two_way_win(board, min_player)
    if trap_col_min is not None:
        if total_pieces < 6:
            return trap_col_min, 8e4 
    trap_col_max = detect_two_way_win(board, max_player)

    if trap_col_max is not None:
        if total_pieces < 6:
            return trap_col_max, 8e4
        
    return iterative_deepening_minimax(board, depth, alpha, beta, max_player, min_player, maximizing_player, time_limit)
