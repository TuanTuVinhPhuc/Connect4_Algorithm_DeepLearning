import math
from functools import lru_cache
import time
import numpy as np
from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
from board import drop_piece, get_next_open_row, winning_move, get_valid_locations, is_terminal_node, is_valid_location

PREFERRED_COLUMNS = [3, 2, 4, 1, 5, 0, 6]

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
        return -5
    elif opp_count == 2 and empty_count == 2:
        return -3

    return 0

def score_position(board, piece):
    score = 0

    # Ưu tiên cột giữa
    center_array = [row[COLUMNS // 2] for row in board]
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

def detect_two_way_win(board, piece, look_ahead=2):
    valid_locations = get_valid_locations(board)
   
   # Kiểm tra nước hai đường thắng ngay lập tức
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

    # Chỉ AI khi tấn công mới xem xét look_ahead
    if look_ahead > 0 and piece == AI:
        first_odd_potential = None
        opponent = PLAYER

        for move in valid_locations:
            row = get_next_open_row(board, move)
            if row is None:
                continue

            # Đi thử nước đầu tiên
            board_copy = board.copy()
            board_copy[row][move] = piece

            # Đếm có bao nhiêu nước đối thủ dẫn đến ta có cơ hội tạo được hai đường thắng
            potential_wins = 0

            # Xét từng nước đi của đối thủ
            for opp_move in get_valid_locations(board_copy):
                opp_row = get_next_open_row(board_copy, opp_move)
                if opp_row is None:
                    continue

                # Đối thủ đi
                board_after_opp = board_copy.copy()
                board_after_opp[opp_row][opp_move] = opponent

                # Thay vì gọi đệ quy, kiểm tra trực tiếp
                for our_next_move in get_valid_locations(board_after_opp):
                    our_next_row = get_next_open_row(board_after_opp, our_next_move)
                    if our_next_row is None:
                        continue

                    final_board = board_after_opp.copy()
                    final_board[our_next_row][our_next_move] = piece

                    # Kiểm tra xem nước đi này có tạo ra hai đường thắng không
                    win_threats = 0
                    for final_next_move in get_valid_locations(final_board):
                        if final_next_move == our_next_move:
                            continue

                        final_next_row = get_next_open_row(final_board, final_next_move)
                        if final_next_row is None:
                            continue

                        check_board = final_board.copy()
                        check_board[final_next_row][final_next_move] = piece

                        if winning_move(check_board, piece)[0]:
                            win_threats += 1
                            if win_threats >= 2:
                                potential_wins += 1
                                break

                    if win_threats >= 2:
                        break

                if potential_wins >= 1:
                    if move % 2 == 0:
                        return move  # Ưu tiên cột chẵn
                    elif first_odd_potential is None:
                        first_odd_potential = move
                    break

        # Nếu không có cột chẵn nhưng có cột lẻ
        if first_odd_potential is not None:
            return first_odd_potential

    return None

def is_symmetric(board):
    """Kiểm tra xem bàn cờ có đối xứng không"""
    for r in range(ROWS):
        for c in range(COLUMNS // 2):
            if board[r][c] != board[r][COLUMNS-1-c]:
                return False
    return True

def get_symmetry_factor(board):
    """Trả về 0.5 nếu bàn cờ đối xứng, 1.0 nếu không"""
    return 0.5 if is_symmetric(board) else 1.0

def filter_symmetric_moves(board, valid_locations):
    """Loại bỏ các nước đi đối xứng trong bàn cờ đối xứng"""
    # Nếu bàn cờ trống hoặc gần như trống (có ít quân), xét đối xứng
    if np.count_nonzero(board) <= 2:
        # Chỉ giữ lại các cột bên phải của cột giữa
        return [col for col in valid_locations if col >= COLUMNS // 2]
    
    # Kiểm tra đối xứng ngang
    symmetric = True
    for r in range(ROWS):
        for c in range(COLUMNS // 2):
            if board[r][c] != board[r][COLUMNS-1-c]:
                symmetric = False
                break
        if not symmetric:
            break
    
    if symmetric:
        # Nếu bàn cờ đối xứng, chỉ xét một nửa các nước đi
        filtered = []
        seen = set()
        
        for col in valid_locations:
            symmetric_col = COLUMNS - 1 - col
            if col >= COLUMNS // 2 or symmetric_col in seen:
                filtered.append(col)
            seen.add(col)
        
        return filtered
    
    return valid_locations

def order_moves(board, valid_locations, maximizing_player, history=None):
    """Sắp xếp nước đi theo thứ tự ưu tiên và loại bỏ đối xứng"""
    if history is None:
        history = {}
    
    # Trước tiên, lọc bỏ các nước đi đối xứng (Symmetry Reduction)
    filtered_locations = filter_symmetric_moves(board, valid_locations)
    
    # Sau đó ưu tiên theo thứ tự cột
    move_scores = []
    
    for col in filtered_locations:
        score = 0
        # Ưu tiên cột trong PREFERRED_COLUMNS
        pref_index = PREFERRED_COLUMNS.index(col) if col in PREFERRED_COLUMNS else 99
        score -= pref_index * 10  # Cột ưu tiên cao có điểm cao hơn
        
        # Ưu tiên cột chẵn
        if col % 2 == 0:
            score += 5
            
        # Nếu cột này có trong history, ưu tiên thêm
        if (col, maximizing_player) in history:
            score += history[(col, maximizing_player)]
            
        # Thử nước đi và đánh giá
        row = get_next_open_row(board, col)
        if row is not None:
            temp_board = board.copy()
            temp_board[row][col] = AI if maximizing_player else PLAYER
            if maximizing_player:
                eval_score = cached_score_position(tuple(map(tuple, temp_board)), AI)
                score += eval_score
            else:
                eval_score = -cached_score_position(tuple(map(tuple, temp_board)), AI)
                score += eval_score
                
        move_scores.append((col, score))
        
    # Sắp xếp theo điểm giảm dần
    move_scores.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in move_scores]

def minimax_with_prunning(board, depth, alpha, beta, maximizing_player, history=None, time_limit=None, start_time=None):
    current_piece = AI 
    opponent_piece = PLAYER 

    win_col = checking_winning(board, current_piece)
    if win_col is not None:
        return win_col, 1e5 

    # Nếu cần chặn đối thủ
    block_col = checking_winning(board, opponent_piece)
    if block_col is not None:
        return block_col, 9e4
    
    if history is None:
        history = {}

    if time_limit and start_time and time.time() - start_time > time_limit:
        # Hết thời gian, trả về kết quả hiện tại
        return None, 0
    
    valid_locations = get_valid_locations(board)
    if not valid_locations:
        return None, 0
    
    is_terminal = is_terminal_node(board)
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI)[0]:  # AI thắng
                return (None, 1e5 + depth)
            elif winning_move(board, PLAYER)[0]:  # Người chơi thắng
                return (None, -1e5 - depth)
            else:  # Hết nước đi (hòa)
                return (None, 0)
        else:  # Đến độ sâu 0
            sym_factor = get_symmetry_factor(board)
            return (None, cached_score_position(tuple(map(tuple, board)), AI) * sym_factor)
        
    # Sắp xếp các nước đi theo thứ tự ưu tiên (Move Ordering)
    ordered_locations = order_moves(board, valid_locations, maximizing_player, history)

    if maximizing_player:  # AI đi
        value = -math.inf
        column = ordered_locations[0] if ordered_locations else None
        
        for col in ordered_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
        
            # Thực hiện nước đi
            board[row][col] = AI
            
            # Đánh giá
            new_score = minimax_with_prunning(board, depth-1, alpha, beta, False, history, time_limit, start_time)[1]
            
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
    else:  # Người chơi đi
        value = math.inf
        column = ordered_locations[0] if ordered_locations else None
        
        for col in ordered_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
        
            # Thực hiện nước đi
            board[row][col] = PLAYER
            
            # Đánh giá
            new_score = minimax_with_prunning(board, depth-1, alpha, beta, True, history, time_limit, start_time)[1]
            
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

# Tối ưu hóa hàm score_position với cache
@lru_cache(maxsize=10000)
def cached_score_position(board_tuple, piece):
    # Chuyển tuple thành array để xử lý
    board = np.array(board_tuple).reshape((ROWS, COLUMNS))
    return score_position(board, piece)

def iterative_deepening_minimax(board, max_depth, alpha, beta, maximizing_player, time_limit=5.0):
    start_time = time.time()
    best_move = None
    best_score = -math.inf if maximizing_player else math.inf
    history = {}  # Lưu lại thông tin nước đi từ các độ sâu trước
    
    # Bắt đầu từ độ sâu 1, tăng dần lên max_depth
    for current_depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit * 0.8:
            # Nếu đã sử dụng 80% thời gian, dừng lại
            break
            
        move, score = minimax_with_prunning(
            board, current_depth, alpha, beta, maximizing_player, 
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

def minimax_v1(board, depth, alpha, beta, maximizing_player, time_limit=5.0):
    if not isinstance(board, np.ndarray):
        board = np.array(board)

    current_piece = AI if maximizing_player else PLAYER
    opponent_piece = PLAYER if maximizing_player else AI

    total_pieces = np.count_nonzero(board != 0)

    if total_pieces >= 6:
        # Nếu có nước thắng ngay
        win_col = checking_winning(board, current_piece)
        if win_col is not None:
            return win_col, 1e5 if maximizing_player else -1e5

        # Nếu cần chặn đối thủ
        block_col = checking_winning(board, opponent_piece)
        if block_col is not None:
            return block_col, 9e4 if maximizing_player else -9e4
        
    trap_col = detect_two_way_win(board, AI)
    if trap_col is not None:
        if total_pieces < 10:
            # Nếu còn ít quân, gài bẫy luôn
            return trap_col, 8e4
        else:
            # Nếu nhiều quân, gài bẫy được ưu tiên thử đầu tiên trong minimax_with_prunning
            valid_locations = get_valid_locations(board)
            if trap_col in valid_locations:
                valid_locations.remove(trap_col)
                valid_locations = [trap_col] + valid_locations

            return minimax_with_prunning(board, depth, alpha, beta, maximizing_player)
        
    return iterative_deepening_minimax(board, depth, alpha, beta, maximizing_player, time_limit)

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

