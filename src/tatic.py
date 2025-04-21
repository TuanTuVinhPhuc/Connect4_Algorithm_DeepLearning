import numpy as np
from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
from board import drop_piece, get_next_open_row, winning_move, get_valid_locations, is_terminal_node, is_valid_location

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


def find_empty_in_three_in_a_row(board, piece):
    result = []
    for r in range(ROWS):
        for c in range(COLUMNS - 3):
            line = [board[r][c+i] for i in range(4)]
            if line.count(piece) == 3 and line.count(0) == 1:
                idx = line.index(0)
                result.append((r, c + idx))

    for r in range(ROWS - 3):
        for c in range(COLUMNS):
            line = [board[r+i][c] for i in range(4)]
            if line.count(piece) == 3 and line.count(0) == 1:
                idx = line.index(0)
                result.append((r + idx, c))

    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            line = [board[r+i][c+i] for i in range(4)]
            if line.count(piece) == 3 and line.count(0) == 1:
                idx = line.index(0)
                result.append((r + idx, c + idx))

    for r in range(3, ROWS):
        for c in range(COLUMNS - 3):
            line = [board[r-i][c+i] for i in range(4)]
            if line.count(piece) == 3 and line.count(0) == 1:
                idx = line.index(0)
                result.append((r - idx, c + idx))

    return result


def find_three_in_row_columns(board, piece):
    result = []
    for col in range(COLUMNS):
        row = get_next_open_row(board, col)
        if row is None:
            continue

        # Copy board và drop quân giả
        board_copy = [r.copy() for r in board]
        board_copy[row][col] = piece

        # Ngang
        for r in range(ROWS):
            for c in range(COLUMNS - 3):
                window = [board_copy[r][c+i] for i in range(4)]
                if window.count(piece) == 3 and window.count(0) == 1:
                    idx = window.index(0)
                    if get_next_open_row(board_copy, c + idx) == r:
                        result.append(col)

        # Dọc
        for c in range(COLUMNS):
            for r in range(ROWS - 3):
                window = [board_copy[r+i][c] for i in range(4)]
                if window.count(piece) == 3 and window.count(0) == 1:
                    idx = window.index(0)
                    if get_next_open_row(board_copy, c) == r + idx:
                        result.append(col)

        # Chéo xuống
        for r in range(ROWS - 3):
            for c in range(COLUMNS - 3):
                window = [board_copy[r+i][c+i] for i in range(4)]
                if window.count(piece) == 3 and window.count(0) == 1:
                    idx = window.index(0)
                    if get_next_open_row(board_copy, c + idx) == r + idx:
                        result.append(col)

        # Chéo lên
        for r in range(3, ROWS):
            for c in range(COLUMNS - 3):
                window = [board_copy[r-i][c+i] for i in range(4)]
                if window.count(piece) == 3 and window.count(0) == 1:
                    idx = window.index(0)
                    if get_next_open_row(board_copy, c + idx) == r - idx:
                        result.append(col)

    return list(set(result))

def find_common_columns(three_in_rows):
    # Tạo dictionary để đếm số lần xuất hiện của mỗi cột
    col_count = {}
    for move_col, (_, empty_col) in three_in_rows:
        if empty_col not in col_count:
            col_count[empty_col] = []
        col_count[empty_col].append(move_col)
    
    # Trả về các cột có nhiều lần xuất hiện (có thể chiến thắng từ nhiều hướng)
    return {col: moves for col, moves in col_count.items() if len(moves) >= 1}

# Xác định người đi trước 
def who_goes_first_in_cols(board, empty_col, turn):
    # Đếm số ô còn trống trong tất cả các cột trừ cột empty_col
    num_fill = 0
    for c in range(COLUMNS):
        if c == empty_col:
            continue
        for r in range(ROWS):
            if board[r][c] == EMPTY:
                num_fill += 1
    
    # Xác định người chơi sau num_fill lượt nữa
    if num_fill % 2 == 0:
        return turn
    else:
        return 3 - turn  # Chuyển đổi giữa 1 và 2
    
def simulate_column_play(board, col, first_player):
    # Mô phỏng lần lượt từng lượt vào cột đó
    board_copy = board.copy()
    turn = first_player
    
    while True:
        if not is_valid_location(board_copy, col):
            break  # cột đã đầy

        row = get_next_open_row(board_copy, col)
        if row is None:
            break
            
        drop_piece(board_copy, row, col, turn)

        if winning_move(board_copy, turn)[0]:
            return turn  # Người này thắng

        turn = 3 - turn  # Chuyển đổi lượt

    return 0  # Hòa nếu không ai thắng

def check_opponent_forced_play(board, piece, avoid_column):
    opponent = PLAYER if piece == AI else AI
    
    for col in range(COLUMNS):
        if col == avoid_column:
            continue
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is None:
                continue

            # Thử cho đối thủ đi
            board[row][col] = opponent
            win = winning_move(board, opponent)[0]
            # Hoàn tác
            board[row][col] = EMPTY

            if win:
                return col
                
    return None

def check_creating_two_threats(board, piece, col_to_avoid): 
    for col in get_valid_locations(board):
        if col == col_to_avoid:
            continue
            
        row = get_next_open_row(board, col)
        if row is None:
            continue
            
        # Thử đi
        board_copy = board.copy()
        board_copy[row][col] = piece
        
        # Kiểm tra số điểm ba có thể tạo ra được
        threat_positions = find_empty_in_three_in_a_row(board_copy, piece)
        
        # Lọc các vị trí có thể chơi được
        playable_threats = []
        for empty_r, empty_c in threat_positions:
            # Chỉ quan tâm đến những vị trí trống mà có thể chơi được
            if empty_r == ROWS - 1 or board_copy[empty_r + 1][empty_c] != EMPTY:
                playable_threats.append((empty_r, empty_c))
        
        # Nếu có từ 2 mối đe dọa trở lên
        if len(playable_threats) >= 2:
            return col
    
    return None
    
def analyze_forced_win_situation(board, piece=AI):
    current_turn = piece  # Giả sử lượt hiện tại là của người gọi hàm
    opp = PLAYER if piece == AI else AI
    
    # Tìm kiếm các nước đi có thể tạo ra thế 3 quân + 1 trống
    potential_moves = find_three_in_row_columns(board, piece)
    if not potential_moves:
        return None, -1, None, 0

    # Xét từng nước đi tiềm năng
    for move_col in potential_moves:
        row = get_next_open_row(board, move_col)
        if row is None:
            continue
            
        # Đi thử nước đó
        board_copy = board.copy()
        board_copy[row][move_col] = piece

        if checking_winning(board_copy, opp) is not None:
            continue
        
        # Tìm các vị trí ô trống
        positions = find_empty_in_three_in_a_row(board_copy, piece)
        
        # Xét từng vị trí trống tiềm năng
        for empty_row, empty_col in positions:
            if not is_valid_location(board_copy, empty_col):
                continue
                
            who_plays_first_in_empty_col = who_goes_first_in_cols(board_copy, empty_col, current_turn)
            player_winning = simulate_column_play(board_copy, empty_col, who_plays_first_in_empty_col)
            
            # Nếu đối thủ sẽ đi ở vị trí dưới ô trống, đây là chiến thuật tốt
            if player_winning == piece:
                # Tạo hai mối đe dọa 
                two_threats_col = check_creating_two_threats(board, piece, empty_col)
                if two_threats_col is not None:
                    return two_threats_col, empty_row, empty_col, 6e4
                
                return move_col, empty_row, empty_col, 7e4
    
    return None, -1, None, 0


def check_opponent_three_in_a_row(board, piece, avoid_column):
    opponent = PLAYER if piece == AI else AI
    
    for r in range(ROWS):
        for c in range(COLUMNS - 3):
            # Kiểm tra hàng ngang
            window = [board[r][c+i] for i in range(4)]
            if window.count(opponent) == 2 and window.count(EMPTY) == 2:
                # Xác định vị trí trống
                empty_indices = [i for i, val in enumerate(window) if val == EMPTY]
                empty_cols = [c + i for i in empty_indices]
                
                # Kiểm tra xem thế này có thể dẫn đến buộc ta đi vào avoid_column không
                if avoid_column in empty_cols:
                    other_col = empty_cols[0] if empty_cols[1] == avoid_column else empty_cols[1]
                    # Kiểm tra xem ô trống đó có thể đi được không
                    row = get_next_open_row(board, other_col)
                    if row is not None and row == r:
                        return other_col, 5e4  # Đi vào cột này để chặn
    
    # Tương tự cho hàng dọc 
    for c in range(COLUMNS):
        for r in range(ROWS - 3):
            window = [board[r+i][c] for i in range(4)]
            if window.count(opponent) == 2 and window.count(EMPTY) == 2:
                empty_indices = [i for i, val in enumerate(window) if val == EMPTY]
                empty_rows = [r + i for i in empty_indices]
                
                # Kiểm tra xem cột này có phải là avoid_column không
                if c == avoid_column and r <= get_next_open_row(board, c) <= r+3:
                    # Đối thủ đang cố tạo 3 quân ở cột ta đang tránh
                    # Tìm một nước để phá kế hoạch của họ
                    for empty_row in empty_rows:
                        if empty_row == get_next_open_row(board, c):
                            return c, 5e4  # Buộc phải đi vào avoid_column để chặn
    
    # Tương tự cho đường chéo xuống
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = [board[r+i][c+i] for i in range(4)]
            if window.count(opponent) == 2 and window.count(EMPTY) == 2:
                empty_indices = [i for i, val in enumerate(window) if val == EMPTY]
                empty_positions = [(r+i, c+i) for i in empty_indices]
                
                for empty_r, empty_c in empty_positions:
                    if empty_c == avoid_column:
                        # Tìm cột khác để chặn
                        other_position = empty_positions[0] if empty_positions[1] == (empty_r, empty_c) else empty_positions[1]
                        other_r, other_c = other_position
                        if other_r == get_next_open_row(board, other_c):
                            return other_c, 5e4
    
    # Tương tự cho đường chéo lên
    for r in range(3, ROWS):
        for c in range(COLUMNS - 3):
            window = [board[r-i][c+i] for i in range(4)]
            if window.count(opponent) == 2 and window.count(EMPTY) == 2:
                empty_indices = [i for i, val in enumerate(window) if val == EMPTY]
                empty_positions = [(r-i, c+i) for i in empty_indices]
                
                for empty_r, empty_c in empty_positions:
                    if empty_c == avoid_column:
                        # Tìm cột khác để chặn
                        other_position = empty_positions[0] if empty_positions[1] == (empty_r, empty_c) else empty_positions[1]
                        other_r, other_c = other_position
                        if other_r == get_next_open_row(board, other_c):
                            return other_c, 5e4
    
    return None, 0

def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))
def main():
    """
    Hàm main đơn giản để test 3 hàm trong tatic.py
    """
    import numpy as np
    from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
    
    # Ma trận từ người dùng
    matrix = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 2, 1, 1, 1],
        [2, 2, 1, 1, 2, 1, 1]
    ]
    
    # Chuyển đổi sang numpy array
    board = np.array(matrix)
    
    print("========== TEST CÁC HÀM CHIẾN THUẬT TRÁNH CỘT ==========")
    print("\nBàn cờ ban đầu:")
    print_board(board)
    
    from tatic import check_opponent_forced_play, check_creating_two_threats, analyze_forced_win_situation
    
    print(check_opponent_three_in_a_row(matrix, 2, 6))
    
    # # Test 1: check_opponent_forced_play
    # print("\n===== Test 1: check_opponent_forced_play =====")
    # avoid_column = 6  # Giả sử cột 6 là cột cần tránh
    # forced_col = check_opponent_forced_play(board, AI, avoid_column)
    # print(f"Kết quả: AI cần đi vào cột {forced_col if forced_col is not None else 'None'} để chặn đối thủ thắng ngay")
    
    # # Test 2: check_creating_two_threats
    # print("\n===== Test 2: check_creating_two_threats =====")
    # two_threats_col = check_creating_two_threats(board, AI, avoid_column)
    # print(f"Kết quả: AI nên đi vào cột {two_threats_col if two_threats_col is not None else 'None'} để tạo hai mối đe dọa")
    
    # # Test 3: analyze_forced_win_situation
    # print("\n===== Test 3: analyze_forced_win_situation =====")
    # move_col, avoid_col, score = analyze_forced_win_situation(board, AI)
    # print(f"Kết quả:")
    # print(f"- Cột cần đi: {move_col if move_col is not None else 'None'}")
    # print(f"- Cột cần tránh: {avoid_col if avoid_col is not None else 'None'}")
    # print(f"- Điểm số: {score}")
    
    # # Kiểm tra với ví dụ khác - đổi vai trò
    # print("\n===== Test với PLAYER =====")
    # move_col, avoid_col, score = analyze_forced_win_situation(board, PLAYER)
    # print(f"Kết quả analyze_forced_win_situation cho PLAYER:")
    # print(f"- Cột cần đi: {move_col if move_col is not None else 'None'}")
    # print(f"- Cột cần tránh: {avoid_col if avoid_col is not None else 'None'}")
    # print(f"- Điểm số: {score}")
    
    # # Test với một ví dụ đơn giản hơn
    # print("\n===== Test với ma trận đơn giản hơn =====")
    # simple_matrix = [
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 2, 2, 0, 0, 0]
    # ]
    # simple_board = np.array(simple_matrix)
    # print("\nBàn cờ đơn giản:")
    # print_board(simple_board)
    
    # move_col, avoid_col, score = analyze_forced_win_situation(simple_board, AI)
    # print(f"Kết quả analyze_forced_win_situation cho AI:")
    # print(f"- Cột cần đi: {move_col if move_col is not None else 'None'}")
    # print(f"- Cột cần tránh: {avoid_col if avoid_col is not None else 'None'}")
    # print(f"- Điểm số: {score}")

if __name__ == "__main__":
    main()