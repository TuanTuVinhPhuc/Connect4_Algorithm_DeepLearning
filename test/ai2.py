import os
import sys
import numpy as np
import math
import random
import time

# Đảm bảo import đúng các module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import các module từ src
try:
    from src.settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
    from src.board import is_valid_location, get_next_open_row, get_valid_locations, create_board, drop_piece, winning_move
except ImportError:
    # Thử cách khác nếu không import được
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
    from board import is_valid_location, get_next_open_row, get_valid_locations, create_board, drop_piece, winning_move

# Hằng số cho MCTS cải tiến
C_PARAM = 1.5  # Tăng tham số exploration để thăm dò nhiều hơn
SIM_TIME = 2.0  # Thời gian tối đa cho MCTS (giây)
MIN_SIMULATIONS = 3000  # Số mô phỏng tối thiểu
ROLLOUT_POLICY_PROB = 0.8  # Xác suất sử dụng chính sách rollout (thay vì ngẫu nhiên)
USE_HEURISTIC = True  # Sử dụng heuristic đánh giá trong rollout

class MCTSNode:
    """Node trong cây MCTS"""
    def __init__(self, board, parent=None, move=None, player=AI):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Nước đi dẫn đến node này
        self.player = player  # Người chơi sẽ đi tiếp theo
        self.children = []  # Các node con
        self.wins = 0  # Số lần thắng
        self.visits = 0  # Số lần thăm node
        self.untried_moves = get_valid_locations(board)  # Các nước đi chưa thử
        self.terminal = False  # Node này có phải là trạng thái kết thúc
        
        # Lưu trữ tổng điểm số để tính giá trị trung bình
        self.total_value = 0.0
        
        # Kiểm tra xem node này có phải là trạng thái kết thúc không
        if winning_move(board, AI)[0] or winning_move(board, PLAYER)[0] or len(self.untried_moves) == 0:
            self.terminal = True
    
    def uct_select_child(self):
        """Chọn node con tốt nhất dựa trên UCT"""
        # Tất cả node con đều phải được thăm ít nhất một lần
        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        
        # Tính giá trị UCT cho tất cả các node con
        log_total_visits = math.log(self.visits)
        
        # Sử dụng hàm max với key là lambda để tìm node con có giá trị UCT cao nhất
        return max(self.children, key=lambda c: c.get_uct_value(log_total_visits))
    
    def get_uct_value(self, log_parent_visits):
        """Tính giá trị UCT với tham số log_parent_visits đã tính sẵn"""
        if self.visits == 0:
            return float('inf')
        
        # Tính toán giá trị trung bình
        avg_value = self.total_value / self.visits
        
        # Thêm thành phần thăm dò
        exploration = C_PARAM * math.sqrt(log_parent_visits / self.visits)
        
        # Cộng thêm một yếu tố nhỏ để ưu tiên các nước đi ở giữa bàn cờ
        center_bias = 0
        if self.move is not None:
            center_bias = (1.0 - abs(self.move - COLUMNS // 2) / (COLUMNS // 2)) * 0.1
        
        return avg_value + exploration + center_bias
    
    def expand(self):
        """Mở rộng node hiện tại bằng cách thêm một node con mới"""
        if not self.untried_moves or self.terminal:
            return None
        
        # Ưu tiên các nước đi ở giữa bàn cờ
        center_col = COLUMNS // 2
        weighted_moves = []
        for move in self.untried_moves:
            # Tính trọng số dựa trên khoảng cách đến cột giữa
            weight = 1.0 + (1.0 - abs(move - center_col) / center_col) * 2.0
            weighted_moves.append((move, weight))
        
        # Chọn một nước đi dựa trên trọng số
        moves, weights = zip(*weighted_moves)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        move = np.random.choice(moves, p=normalized_weights)
        
        self.untried_moves.remove(move)
        
        # Tạo bảng mới dựa trên nước đi này
        child_board = self.board.copy()
        row = get_next_open_row(child_board, move)
        if row is not None:
            # Đặt quân của người chơi hiện tại
            drop_piece(child_board, row, move, self.player)
            
            # Tạo node con với người chơi đối phương ở lượt tiếp
            next_player = PLAYER if self.player == AI else AI
            child_node = MCTSNode(child_board, self, move, next_player)
            self.children.append(child_node)
            
            return child_node
        
        return None
    
    def update(self, result):
        """Cập nhật số lần thăm và tổng điểm"""
        self.visits += 1
        self.total_value += result
    
    def is_fully_expanded(self):
        """Kiểm tra xem tất cả các nước đi có thể đã được thử chưa"""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """Kiểm tra xem node này có phải là trạng thái kết thúc không"""
        return self.terminal
    
    def get_best_move(self):
        """Lấy nước đi tốt nhất dựa trên số lần thăm"""
        if not self.children:
            return None
        
        # Chọn node con có số lần thăm nhiều nhất
        return max(self.children, key=lambda c: c.visits).move
    
    def get_most_promising_child(self):
        """Lấy node con hứa hẹn nhất dựa trên tỷ lệ thắng"""
        if not self.children:
            return None
        
        # Chọn node con có tỷ lệ thắng cao nhất
        return max(self.children, key=lambda c: c.total_value / max(c.visits, 1))

def evaluate_board(board, piece):
    """Đánh giá trạng thái bảng dựa trên heuristic"""
    score = 0
    opp_piece = PLAYER if piece == AI else AI
    
    # Đánh giá các cửa sổ 4 ô
    # Hàng ngang
    for r in range(ROWS):
        for c in range(COLUMNS-3):
            window = [board[r][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Hàng dọc
    for c in range(COLUMNS):
        for r in range(ROWS-3):
            window = [board[r+i][c] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Đường chéo xuống
    for r in range(ROWS-3):
        for c in range(COLUMNS-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Đường chéo lên
    for r in range(3, ROWS):
        for c in range(COLUMNS-3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Ưu tiên cột giữa
    center_array = [int(i) for i in list(board[:, COLUMNS//2])]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    return score

def evaluate_window(window, piece):
    """Đánh giá cửa sổ 4 ô"""
    opp_piece = PLAYER if piece == AI else AI
    
    # Đếm nhanh số lượng mỗi loại quân trong cửa sổ
    piece_count = window.count(piece)
    empty_count = window.count(EMPTY)
    opp_count = window.count(opp_piece)
    
    # Tính điểm dựa trên các trường hợp
    if piece_count == 4:
        return 100
    elif piece_count == 3 and empty_count == 1:
        return 5
    elif piece_count == 2 and empty_count == 2:
        return 2
    elif opp_count == 3 and empty_count == 1:
        return -4
    
    return 0

def rollout_policy(board, player):
    """
    Chính sách rollout thông minh hơn ngẫu nhiên:
    - Ưu tiên nước thắng
    - Chặn nước thua
    - Ưu tiên cột giữa
    - Lựa chọn dựa trên heuristic
    """
    valid_moves = get_valid_locations(board)
    if not valid_moves:
        return None
    
    # Kiểm tra nước thắng ngay lập tức
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
        
        b_copy = board.copy()
        drop_piece(b_copy, row, move, player)
        if winning_move(b_copy, player)[0]:
            return move
    
    # Chặn nước thắng của đối thủ
    opponent = PLAYER if player == AI else AI
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
        
        b_copy = board.copy()
        drop_piece(b_copy, row, move, opponent)
        if winning_move(b_copy, opponent)[0]:
            return move
    
    # Ưu tiên cột giữa
    if COLUMNS // 2 in valid_moves:
        return COLUMNS // 2
    
    if USE_HEURISTIC:
        # Đánh giá dựa trên heuristic
        move_scores = []
        for move in valid_moves:
            row = get_next_open_row(board, move)
            if row is None:
                continue
            
            b_copy = board.copy()
            drop_piece(b_copy, row, move, player)
            score = evaluate_board(b_copy, player)
            move_scores.append((move, score))
        
        # Chọn một trong 3 nước đi tốt nhất
        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_moves = [move for move, _ in move_scores[:3]]
        if top_moves:
            return random.choice(top_moves)
    
    # Fallback: chọn ngẫu nhiên
    return random.choice(valid_moves)

def rollout(board, player):
    """
    Chạy một mô phỏng từ trạng thái board với người chơi player đi tiếp theo
    """
    current_board = board.copy()
    current_player = player
    depth = 0
    max_depth = ROWS * COLUMNS  # Chiều dài tối đa có thể của game
    
    # Mô phỏng cho đến khi game kết thúc
    while depth < max_depth:
        # Kiểm tra điều kiện kết thúc
        if winning_move(current_board, AI)[0]:
            return 1.0  # AI thắng
        elif winning_move(current_board, PLAYER)[0]:
            return 0.0  # Player thắng
        
        # Lấy tất cả các nước đi hợp lệ
        valid_moves = get_valid_locations(current_board)
        if not valid_moves:
            return 0.5  # Hòa
        
        # Sử dụng policy với xác suất cao, ngẫu nhiên với xác suất thấp
        if random.random() < ROLLOUT_POLICY_PROB:
            move = rollout_policy(current_board, current_player)
        else:
            move = random.choice(valid_moves)
        
        if move is None:
            move = random.choice(valid_moves)
            
        row = get_next_open_row(current_board, move)
        if row is not None:
            drop_piece(current_board, row, move, current_player)
            
            # Chuyển lượt
            current_player = PLAYER if current_player == AI else AI
            depth += 1
        else:
            break
    
    # Nếu đạt đến độ sâu tối đa, đánh giá bảng
    ai_score = evaluate_board(current_board, AI)
    player_score = evaluate_board(current_board, PLAYER)
    
    # Chuẩn hóa điểm số thành giá trị từ 0 đến 1
    total_score = abs(ai_score) + abs(player_score)
    if total_score == 0:
        return 0.5
    
    normalized_score = (ai_score + total_score) / (2 * total_score)
    return min(max(normalized_score, 0.0), 1.0)  # Đảm bảo điểm nằm trong [0, 1]



def mcts_search(board, player=AI):
    """
    Thực hiện tìm kiếm MCTS cho trạng thái bảng hiện tại
    """
    # Kiểm tra nhanh các trường hợp đặc biệt
    # 1. Kiểm tra thắng ngay lập tức
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is not None:
                board_copy = board.copy()
                drop_piece(board_copy, row, col, player)
                if winning_move(board_copy, player)[0]:
                    return col
    
    # 2. Chặn đối thủ thắng
    opponent = PLAYER if player == AI else AI
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is not None:
                board_copy = board.copy()
                drop_piece(board_copy, row, col, opponent)
                if winning_move(board_copy, opponent)[0]:
                    return col
    
    # 3. Nếu bảng trống, ưu tiên cột giữa
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
    
    # 4. Nếu là nước đi đầu tiên và cột giữa trống
    if board[ROWS-1, COLUMNS//2] == EMPTY:
        return COLUMNS // 2
    
    # 5. Nếu không, sử dụng MCTS
    root = MCTSNode(board, player=player)
    
    # Nếu chỉ có một nước đi hợp lệ, chọn nước đó
    if len(root.untried_moves) == 1:
        return root.untried_moves[0]
    
    # Bắt đầu đếm thời gian
    start_time = time.time()
    sim_count = 0
    
    # Chạy MCTS trong thời gian cho phép và số mô phỏng tối thiểu
    while (time.time() - start_time < SIM_TIME or sim_count < MIN_SIMULATIONS):
        # 1. Selection: chọn node tốt nhất từ gốc đến lá
        node = root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.uct_select_child()
        
        # 2. Expansion: mở rộng node lá nếu nó không phải là node kết thúc
        if not node.is_terminal():
            new_node = node.expand()
            if new_node:
                node = new_node
        
        # 3. Simulation: chạy mô phỏng từ node hiện tại
        result = rollout(node.board, node.player)
        
        # 4. Backpropagation: cập nhật thông tin lên các node cha
        while node:
            # Cập nhật kết quả tùy thuộc vào lượt chơi của node
            if node.player == player:
                # Nếu AI (hoặc người chơi hiện tại) đi tiếp theo, node cha phải là
                # node đi trước đó của đối thủ, vì vậy giá trị cần phải đảo ngược
                node.update(1 - result)
            else:
                node.update(result)
            node = node.parent
        
        sim_count += 1
        
        # Nếu đã tìm thấy nước thắng chắc chắn, dừng sớm
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        if best_child and best_child.visits > 10 and best_child.total_value / best_child.visits > 0.95:
            break
    
    # In thông tin debug
    print(f"MCTS: {sim_count} mô phỏng trong {time.time() - start_time:.3f}s")
    if root.children:
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
            if child.visits > 0:
                win_rate = child.total_value / child.visits
                print(f"  Cột {child.move}: {child.visits} lần thăm, tỷ lệ thắng: {win_rate:.3f}")
    
    # Chọn nước đi tốt nhất dựa trên số lần thăm
    return root.get_best_move()

def random_move(board):
    """
    Hàm interface cho ai_battle.py, trả về nước đi tối ưu sử dụng MCTS
    """
    return mcts_search(board)



if __name__ == "__main__":
    # Tạo bảng trống
    test_board = create_board()
    
    # Thử một số nước đi
    drop_piece(test_board, 5, 3, PLAYER)
    drop_piece(test_board, 5, 2, AI)
    drop_piece(test_board, 4, 3, PLAYER)
    
    # In bảng
    print("Bảng test:")
    print(test_board)
    
    # Chạy MCTS đa luồng
    start = time.time()
    move = mcts_search(test_board)
    end = time.time()
    
    print(f"MCTS đa luồng chọn cột: {move} (Thời gian: {end-start:.4f}s)")


