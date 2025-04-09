import math
import numpy as np
from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
from board import drop_piece, get_next_open_row, winning_move, get_valid_locations, is_terminal_node, is_valid_location

def checking_winning(board, piece):
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)

            if row is None:
                continue
            b_copy = board.copy()
            drop_piece(b_copy, row, col, piece)

            if (winning_move(b_copy, piece)[0]):
                return col
    return None

def evaluate_window(window, piece):
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

    # Phòng thủ với người chơi
    if opp_count == 3 and empty_count == 1:
        return -6
    elif opp_count == 2 and empty_count == 2:
        return -3

    return 0

def score_position(board, piece):
    score = 0
    opp_piece = PLAYER if piece == AI else AI

    # Ưu tiên cột giữa
    center_array = [int(i) for i in list(board[:, COLUMNS//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Điểm hàng ngang
    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMNS-3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    # Điểm hàng dọc
    for c in range(COLUMNS):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROWS-3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    # Điểm đường chéo xuống
    for r in range(ROWS-3):
        for c in range(COLUMNS-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Điểm đường chéo lên
    for r in range(3, ROWS):
        for c in range(COLUMNS-3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

def minimax_with_checks(board, depth, alpha, beta, maximizing_player):
    valid_locations = get_valid_locations(board)
    if not valid_locations:
        return None, 0
    
    is_terminal = is_terminal_node(board)
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI)[0]:  # AI thắng
                return (None, 100)
            elif winning_move(board, PLAYER)[0]:  # Người chơi thắng
                return (None, -100)
            else:  # Hết nước đi (hòa)
                return (None, 0)
        else:  # Đến độ sâu 0
            return (None, score_position(board, AI))
        
    if depth > 3:
        col_scores = []
        for col in valid_locations:
            row = get_next_open_row(board, col)
            middle_preference = 3 - abs(col - COLUMNS//2)
            col_scores.append((col, middle_preference))
    
        if maximizing_player:
            col_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            col_scores.sort(key=lambda x: x[1])
            
        valid_locations = [c[0] for c in col_scores]
    
    if maximizing_player:  # AI đi
        value = -math.inf
        column = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
        
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI)
            new_score = minimax_with_checks(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:  # Người chơi đi
        value = math.inf
        column = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
        
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER)
            new_score = minimax_with_checks(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value
    
def minimax(board, depth, alpha, beta, maximizing_player):
    if maximizing_player:
        winning_move_col = checking_winning(board, AI)
        if winning_move_col is not None:
            return winning_move_col, 10000
        
    blocking_col = checking_winning(board, PLAYER)
    if blocking_col is not None:
        return blocking_col, 10000
    
    return minimax_with_checks(board, depth, alpha, beta, maximizing_player)

# Lựa chọn cuối cùng 
def pick_best_move(board, piece):
    winning_col = checking_winning(board, piece)
    if winning_col is not None:
        return winning_col
    
    blocking_col = checking_winning(board, PLAYER if piece == AI else AI)
    if blocking_col is not None:
        return blocking_col
    
    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = np.random.choice(valid_locations)
    
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col
            
    return best_col


