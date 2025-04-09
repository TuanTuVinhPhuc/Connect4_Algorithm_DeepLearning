import numpy as np
import math
import random
import time

from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
from board import is_valid_location, get_next_open_row, get_valid_locations, create_board, drop_piece, winning_move

C_PARAM = math.sqrt(2) 
MIN_SIMULATIONS = 30000
ROLLOUT_POLICY_PROB = 0.8  

class MCTSNode:
    def __init__(self, board, parent=None, move=None, player=AI):
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
            # return max(unvisited, key=lambda c: c.heuristic_value + 0.03 * random.random())
        
        # Tính giá trị UCT cho tất cả các node con
        log_total_visits = math.log(self.visits)
        
        # Tìm giá trị UCT cao nhất
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

def evaluate_board(board, piece):
    score = 0

    # center_array = [int(i) for i in list(board[:, COLUMNS//2])]
    center_array = [board[r][COLUMNS//2] for r in range(ROWS)]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    for r in range(ROWS):
        for c in range(COLUMNS-3):
            window = [board[r][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    for c in range(COLUMNS):
        for r in range(ROWS-3):
            window = [board[r+i][c] for i in range(4)]
            score += evaluate_window(window, piece)
    
    for r in range(ROWS-3):
        for c in range(COLUMNS-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    

    for r in range(3, ROWS):
        for c in range(COLUMNS-3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    return score

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
    
    if opp_count == 3 and empty_count == 1:
        return -6
    elif opp_count == 2 and empty_count == 2:
        return -3
    
    return 0

def detect_two_way_win(board, player):
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
            return move, win_threats
            
    return None, 0

def find_forced_win(board, player, depth):
    if depth <= 0:
        return None
        
    valid_moves = get_valid_locations(board)
    opponent = PLAYER if player == AI else AI
    
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, player)
        
        if winning_move(b_copy, player)[0]:
            return move
            
    two_way_move, threat_count = detect_two_way_win(board, player)
    if two_way_move is not None:
        return two_way_move

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
            return move
            
    return None

def rollout_policy(board, player):
    valid_moves = get_valid_locations(board)
    if not valid_moves:
        return None

    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, player)
        
        if winning_move(b_copy, player)[0]:
            return move
            
    opponent = PLAYER if player == AI else AI
    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, opponent)
        
        if winning_move(b_copy, opponent)[0]:
            return move

    two_way_move, _ = detect_two_way_win(board, player)
    if two_way_move is not None:
        return two_way_move

    opponent_two_way, _ = detect_two_way_win(board, opponent)
    if opponent_two_way is not None:
        return opponent_two_way
    
    forced_win = find_forced_win(board, player, 3)
    if forced_win is not None:
        return forced_win
    
    move_scores = []

    for move in valid_moves:
        row = get_next_open_row(board, move)
        if row is None:
            continue
            
        b_copy = [r[:] for r in board]
        drop_piece(b_copy, row, move, player)
        
        score = evaluate_board(b_copy, player)
        
        center_preference = (1.0 - abs(move - COLUMNS // 2) / (COLUMNS // 2)) * 2
        score += center_preference
        
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
        
    if COLUMNS // 2 in valid_moves:
        return COLUMNS // 2

    return random.choice(valid_moves)

def neural_rollout(board, player, neural_network):
    # Dự đoán giá trị từ neural network
    _, value = neural_network.predict(board)
    
    if player == AI:
        return value
    else:
        return 1.0 - value
    
def rollout(board, player, neural_network=None, nn_prob=0.7):
    if neural_network and random.random() < nn_prob:
        return neural_rollout(board, player, neural_network)

    current_board = board.copy()
    current_player = player
    depth = 0
    max_depth = ROWS * COLUMNS  
    
    while depth < max_depth:
        if winning_move(current_board, AI)[0]:
            return 1.0  # AI thắng
        elif winning_move(current_board, PLAYER)[0]:
            return 0.0  # Player thắng
        
        valid_moves = get_valid_locations(current_board)
        if not valid_moves:
            return 0.5  # Hòa
        
        if random.random() < ROLLOUT_POLICY_PROB:
            move = rollout_policy(current_board, current_player)
        else:
            move = random.choice(valid_moves)
        
        if move is None:
            move = random.choice(valid_moves)
            
        row = get_next_open_row(current_board, move)
        if row is not None:
            drop_piece(current_board, row, move, current_player)
            
            current_player = PLAYER if current_player == AI else AI
            depth += 1
        else:
            break
    
    ai_score = evaluate_board(current_board, AI)
    player_score = evaluate_board(current_board, PLAYER)
    
    total_score = abs(ai_score) + abs(player_score)
    if total_score == 0:
        return 0.5
    
    normalized_score = (ai_score + total_score) / (2 * total_score)
    return min(max(normalized_score, 0.0), 1.0)  



def mcts_search(board, player=AI, neural_network=None, simulations=MIN_SIMULATIONS):
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is not None:
                board_copy = board.copy()
                drop_piece(board_copy, row, col, player)
                if winning_move(board_copy, player)[0]:
                    policy = np.zeros(COLUMNS)
                    policy[col] = 1.0
                    return col, policy
    
    opponent = PLAYER if player == AI else AI
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is not None:
                board_copy = board.copy()
                drop_piece(board_copy, row, col, opponent)
                if winning_move(board_copy, opponent)[0]:
                    policy = np.zeros(COLUMNS)
                    policy[col] = 1.0
                    return col, policy
    
    # Ưu tiên cột giữa
    is_empty_board = True
    for r in range(ROWS):
        for c in range(COLUMNS):
            if board[r, c] != EMPTY:
                is_empty_board = False
                break
        if not is_empty_board:
            break
            
    if is_empty_board:
        policy = np.zeros(COLUMNS)
        policy[COLUMNS // 2] = 1.0
        return COLUMNS // 2, policy
    
    if board[ROWS-1, COLUMNS//2] == EMPTY:
        policy = np.zeros(COLUMNS)
        policy[COLUMNS // 2] = 1.0
        return COLUMNS // 2, policy
    
    root = MCTSNode(board, player=player)
    
    if len(root.untried_moves) == 1:
        policy = np.zeros(COLUMNS)
        policy[root.untried_moves[0]] = 1.0
        return root.untried_moves[0], policy

    start_time = time.time()
    sim_count = 0
    
    while (sim_count < simulations):
        node = root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.uct_select_child()
        
        if not node.is_terminal():
            new_node = node.expand()
            if new_node:
                node = new_node
        
        result = rollout(node.board, node.player, neural_network)
        
        while node:
            if node.player == player:
                node.update(1 - result)
            else:
                node.update(result)
            node = node.parent
        
        sim_count += 1
        
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        if best_child and best_child.visits > 100 and best_child.total_value / best_child.visits > 0.95:
            break
    
    print(f"MCTS: {sim_count} mô phỏng trong {time.time() - start_time:.3f}s")

    policy = np.zeros(COLUMNS)
    total_visits = sum(child.visits for child in root.children)

    if root.children:
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
            if child.visits > 0:
                win_rate = child.total_value / child.visits
                print(f"  Cột {child.move}: {child.visits} lần thăm, tỷ lệ thắng: {win_rate:.3f}")
                policy[child.move] = child.visits / total_visits if total_visits > 0 else 0

    return int(root.get_best_move()), policy

def random_move(board):
    return mcts_search(board)

if __name__ == "__main__":
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
    move, mcts_policy = mcts_search(test_board)
    end = time.time()
    
    print(f"MCTS chọn cột: {move} (Thời gian: {end-start:.4f}s)")


