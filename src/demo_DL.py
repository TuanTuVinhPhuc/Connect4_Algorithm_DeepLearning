
# Cell 1: Import các thư viện cần thiết
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, BatchNormalization, Activation, add, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import random
import time
from tqdm.notebook import tqdm

# Import từ game Connect4
from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
from board import is_valid_location, get_next_open_row, get_valid_locations, create_board, drop_piece, winning_move
from MCTS import mcts_search, rollout, rollout_policy, detect_two_way_win, find_forced_win, evaluate_board, MCTSNode

# Kiểm tra và cấu hình GPU nếu có
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Đang sử dụng GPU")
except:
    print("Không tìm thấy GPU, sử dụng CPU")

# Tạo thư mục models nếu chưa tồn tại
os.makedirs('models', exist_ok=True)

# Cell 2: Định nghĩa class Connect4NeuralNetwork tối ưu
class Connect4NeuralNetwork:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        
        # Cache cho dự đoán
        self._prediction_cache = {}
        
        # Cờ báo hiệu model đã sẵn sàng
        self.model_ready = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Xây dựng mạng neural network nhỏ gọn và hiệu quả hơn"""
        input_shape = (ROWS, COLUMNS, 3)
        inputs = Input(shape=input_shape)
        
        # Mạng CNN đơn giản hơn
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Chỉ sử dụng 1 residual block
        x = self._residual_block(x, 32)
        
        # Policy head - dự đoán phân phối xác suất của các nước đi
        policy_head = Conv2D(16, (1, 1), padding='same')(x)
        policy_head = BatchNormalization()(policy_head)
        policy_head = Activation('relu')(policy_head)
        policy_head = Flatten()(policy_head)
        policy_head = Dense(COLUMNS, activation='softmax', name='policy')(policy_head)
        
        # Value head - đánh giá vị thế của bàn cờ
        value_head = Conv2D(16, (1, 1), padding='same')(x)
        value_head = BatchNormalization()(value_head)
        value_head = Activation('relu')(value_head)
        value_head = Flatten()(value_head)
        value_head = Dense(32, activation='relu')(value_head)
        value_head = Dropout(0.3)(value_head)
        value_head = Dense(1, activation='tanh', name='value')(value_head)
        
        # Xây dựng model
        self.model = Model(inputs=inputs, outputs=[policy_head, value_head])
        
        # Biên dịch model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            },
            metrics={
                'policy': 'accuracy',
                'value': 'mean_absolute_error'
            }
        )
        
        # Hiển thị tóm tắt model
        self.model.summary()
        self.model_ready = True
    
    def _residual_block(self, x, filters):
        """Tạo một residual block"""
        shortcut = x
        
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        
        x = add([x, shortcut])
        x = Activation('relu')(x)
        
        return x
    
    def _board_to_key(self, board):
        """Tạo khóa hash cho bàn cờ"""
        key = []
        for r in range(ROWS):
            for c in range(COLUMNS):
                key.append(str(board[r][c]))
        return "".join(key)
    
    def _board_to_tensor(self, board):
        """Chuyển đổi bàn cờ thành tensor 3 kênh"""
        tensor = np.zeros((ROWS, COLUMNS, 3), dtype=np.float32)
        
        for r in range(ROWS):
            for c in range(COLUMNS):
                if board[r][c] == AI:
                    tensor[r, c, 0] = 1
                elif board[r][c] == PLAYER:
                    tensor[r, c, 1] = 1
                else:  # EMPTY
                    tensor[r, c, 2] = 1
        
        return tensor
    
    def predict(self, board):
        """Dự đoán policy và value với cache"""
        if not self.model_ready:
            # Default policy và value nếu model chưa sẵn sàng
            default_policy = np.ones(COLUMNS) / COLUMNS
            default_value = 0.0
            return default_policy, default_value
        
        # Tạo khóa cho cache
        board_key = self._board_to_key(board)
        
        # Kiểm tra cache
        if board_key in self._prediction_cache:
            return self._prediction_cache[board_key]
        
        # Dự đoán với neural network
        board_tensor = self._board_to_tensor(board)
        policy, value = self.model.predict(np.expand_dims(board_tensor, axis=0), verbose=0)
        
        # Lấy ra kết quả
        result = (policy[0], value[0][0])
        
        # Lưu vào cache
        self._prediction_cache[board_key] = result
        
        return result
    
    def batch_predict(self, boards):
        """Dự đoán hàng loạt cho nhiều bàn cờ"""
        if not self.model_ready:
            return None
            
        # Chuẩn bị batch
        batch_tensors = []
        board_keys = []
        uncached_indices = []
        results = [None] * len(boards)
        
        # Kiểm tra cache và chuẩn bị batch
        for i, board in enumerate(boards):
            board_key = self._board_to_key(board)
            board_keys.append(board_key)
            
            if board_key in self._prediction_cache:
                results[i] = self._prediction_cache[board_key]
            else:
                batch_tensors.append(self._board_to_tensor(board))
                uncached_indices.append(i)
        
        # Nếu có bàn cờ chưa trong cache
        if batch_tensors:
            batch_tensors = np.array(batch_tensors)
            policies, values = self.model.predict(batch_tensors, verbose=0)
            
            # Lưu kết quả vào cache và cập nhật results
            for idx, i in enumerate(uncached_indices):
                result = (policies[idx], values[idx][0])
                self._prediction_cache[board_keys[i]] = result
                results[i] = result
        
        return results
    
    def clear_cache(self):
        """Xóa cache"""
        self._prediction_cache = {}
    
    def train(self, boards, policies, values, epochs=50, batch_size=64, validation_split=0.2):
        """Huấn luyện mạng neural network với dữ liệu"""
        # Chuyển đổi dữ liệu đầu vào
        X = np.array([self._board_to_tensor(board) for board in boards])
        y_policy = np.array(policies)
        y_value = np.array(values).reshape(-1, 1)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath='models/best_connect4_model.h5',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Huấn luyện model
        history = self.model.fit(
            X, 
            {'policy': y_policy, 'value': y_value},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Xóa cache vì model đã thay đổi
        self.clear_cache()
        
        return history
    
    def save_model(self, path='models/connect4_model.h5'):
        """Lưu model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model đã được lưu tại {path}")
    
    def load_model(self, path):
        """Tải model đã lưu"""
        self.model = tf.keras.models.load_model(path)
        print(f"Model đã được tải từ {path}")
        self.model_ready = True
    
    def generate_training_data(self, mcts_games, augment=True):
        """
        Tạo dữ liệu huấn luyện từ các trò chơi MCTS
        """
        boards = []
        policies = []
        values = []
        
        for board, policy, result in mcts_games:
            boards.append(board)
            policies.append(policy)
            values.append(result)
            
            # Data augmentation bằng cách lật ngang bàn cờ
            if augment:
                # Tạo bản sao bằng cách lật ngang
                flipped_board = np.flip(board, axis=1).copy()
                flipped_policy = np.flip(policy).copy()
                
                boards.append(flipped_board)
                policies.append(flipped_policy)
                values.append(result)
        
        return boards, policies, values

# Cell 3: Định nghĩa class Connect4Agent tối ưu
class Connect4Agent:
    def __init__(self, neural_network=None, model_path=None):
        """
        Connect4Agent sử dụng neural network để đánh giá bàn cờ và đưa ra quyết định
        """
        if neural_network:
            self.nn = neural_network
        else:
            self.nn = Connect4NeuralNetwork(model_path)
        
        self.temperature = 1.0  # Điều chỉnh mức độ khám phá
        self.batch_size = 32    # Kích thước batch cho dự đoán hàng loạt
    
    def get_move(self, board, valid_moves, exploration=True):
        """
        Chọn nước đi dựa trên mạng neural network
        """
        if not valid_moves:
            return None
        
        # Dự đoán policy và value
        policy, value = self.nn.predict(board)
        
        # Lọc ra chỉ những nước đi hợp lệ
        valid_policy = np.zeros(COLUMNS)
        for move in valid_moves:
            valid_policy[move] = policy[move]
        
        # Nếu không có nước đi nào hợp lệ có xác suất > 0
        if np.sum(valid_policy) == 0:
            # Trả về nước đi ngẫu nhiên từ danh sách nước đi hợp lệ
            return np.random.choice(valid_moves)
        
        # Chuẩn hóa lại policy
        valid_policy /= np.sum(valid_policy)
        
        if exploration:
            # Thêm yếu tố khám phá bằng cách sử dụng softmax với temperature
            valid_policy = np.exp(np.log(valid_policy + 1e-10) / self.temperature)
            valid_policy /= np.sum(valid_policy)
            
            # Chọn nước đi theo xác suất
            move = np.random.choice(COLUMNS, p=valid_policy)
        else:
            # Chọn nước đi có xác suất cao nhất
            move = np.argmax(valid_policy)
        
        return move
    
    def fast_mcts(self, board, valid_moves, num_simulations=500):
        """
        MCTS tối ưu hóa với batch inference và dừng sớm
        """
        # Kiểm tra nhanh các nước đi đặc biệt
        
        # 1. Nước thắng ngay lập tức
        for move in valid_moves:
            row = get_next_open_row(board, move)
            if row is None:
                continue
                
            board_copy = [r[:] for r in board]
            drop_piece(board_copy, row, move, AI)
            if winning_move(board_copy, AI)[0]:
                # One-hot policy (100% chọn nước này)
                policy = np.zeros(COLUMNS)
                policy[move] = 1.0
                return move, policy
                
        # 2. Ngăn chặn đối thủ thắng ngay lập tức
        for move in valid_moves:
            row = get_next_open_row(board, move)
            if row is None:
                continue
                
            board_copy = [r[:] for r in board]
            drop_piece(board_copy, row, move, PLAYER)
            if winning_move(board_copy, PLAYER)[0]:
                # One-hot policy (100% chọn nước này)
                policy = np.zeros(COLUMNS)
                policy[move] = 1.0
                return move, policy
                
        # Tạo root node
        root = MCTSNode(board, player=AI)
        if len(root.untried_moves) == 1:
            # Chỉ có 1 nước đi hợp lệ
            move = root.untried_moves[0]
            policy = np.zeros(COLUMNS)
            policy[move] = 1.0
            return move, policy
        
        # Theo dõi thời gian
        start_time = time.time()
        batch_nodes = []
        sim_count = 0
        max_time = 5.0  # Tối đa 5 giây
        
        while sim_count < num_simulations and (time.time() - start_time) < max_time:
            # Thu thập batch nodes để đánh giá
            nodes_batch = []
            
            # Thực hiện selection/expansion cho batch_size node
            batch_size = min(self.batch_size, num_simulations - sim_count)
            for _ in range(batch_size):
                node = root
                # Selection
                while node.is_fully_expanded() and not node.is_terminal():
                    node = node.uct_select_child()
                
                # Expansion
                if not node.is_terminal():
                    new_node = node.expand()
                    if new_node:
                        node = new_node
                
                nodes_batch.append(node)
                sim_count += 1
            
            # Đánh giá batch nodes bằng neural network
            boards_to_evaluate = [node.board for node in nodes_batch]
            evaluation_results = []
            
            # Đánh giá các node terminal trước
            for node in nodes_batch:
                if node.is_terminal():
                    if winning_move(node.board, AI)[0]:
                        evaluation_results.append(1.0)
                    elif winning_move(node.board, PLAYER)[0]:
                        evaluation_results.append(0.0)
                    else:
                        evaluation_results.append(0.5)  # Hòa
                else:
                    # Sử dụng neural network đánh giá
                    _, value = self.nn.predict(node.board)
                    evaluation_results.append(value)
            
            # Backpropagation cho tất cả các node
            for node, value in zip(nodes_batch, evaluation_results):
                current_node = node
                while current_node:
                    current_node.visits += 1
                    if current_node.player == AI:
                        current_node.total_value += value
                    else:
                        current_node.total_value += (1 - value)
                    current_node = current_node.parent
        
        # Tạo policy từ số lần thăm
        mcts_policy = np.zeros(COLUMNS)
        total_visits = sum(child.visits for child in root.children)
        
        for child in root.children:
            mcts_policy[child.move] = child.visits / total_visits
            
        # Chọn nước đi tốt nhất
        best_move = root.get_best_move()
        
        # In thông tin
        elapsed = time.time() - start_time
        print(f"Fast MCTS: {sim_count} mô phỏng trong {elapsed:.3f}s ({sim_count/elapsed:.1f} mô phỏng/giây)")
        
        # Log các nước đi hàng đầu
        top_moves = sorted(root.children, key=lambda c: c.visits, reverse=True)[:3]
        for child in top_moves:
            if child.visits > 0:
                win_rate = child.total_value / child.visits
                print(f"  Cột {child.move}: {child.visits} lần thăm, tỷ lệ thắng: {win_rate:.3f}")
        
        return best_move, mcts_policy

# Cell 4: Hàm thu thập dữ liệu tối ưu
def collect_self_play_data(agent, num_games=20, mcts_simulations=500):
    """Thu thập dữ liệu từ tự chơi kết hợp MCTS và Neural Network (tối ưu)"""
    training_data = []
    win_stats = {'AI': 0, 'PLAYER': 0, 'DRAW': 0}
    
    for game_idx in tqdm(range(num_games), desc="Self-play games"):
        board = create_board()
        game_memory = []
        current_player = AI if game_idx % 2 == 0 else PLAYER  # Luân phiên người đi trước
        
        game_start_time = time.time()
        
        while True:
            valid_moves = get_valid_locations(board)
            
            if not valid_moves:  # Bàn cờ đầy, hòa
                result = 0.0
                win_stats['DRAW'] += 1
                break
                
            # Thực hiện nước đi
            if current_player == AI:
                # AI sử dụng MCTS + Neural Network
                move, mcts_policy = agent.fast_mcts(board, valid_moves, mcts_simulations)
                
                # Lưu trạng thái bàn cờ và policy
                game_memory.append((board.copy(), mcts_policy, AI))
            else:
                # PLAYER cũng sử dụng MCTS + Neural Network nhưng đóng vai trò đối thủ
                move, mcts_policy = agent.fast_mcts(board, valid_moves, mcts_simulations)
                
                # Lưu trạng thái bàn cờ và policy
                game_memory.append((board.copy(), mcts_policy, PLAYER))
            
            # Thực hiện nước đi
            row = get_next_open_row(board, move)
            drop_piece(board, row, move, current_player)
            
            # Kiểm tra thắng/thua
            if winning_move(board, current_player)[0]:
                if current_player == AI:
                    result = 1.0
                    win_stats['AI'] += 1
                else:
                    result = -1.0
                    win_stats['PLAYER'] += 1
                break
                
            # Chuyển lượt
            current_player = PLAYER if current_player == AI else AI
        
        # Lưu kết quả vào training data
        for board_state, policy, player in game_memory:
            if player == AI:
                training_data.append((board_state, policy, result))
            else:
                # Đối với PLAYER, kết quả ngược lại
                training_data.append((board_state, policy, -result))
        
        # In kết quả và thời gian của trò chơi
        game_time = time.time() - game_start_time
        print(f"Game {game_idx+1} hoàn thành trong {game_time:.1f}s: {'AI' if result > 0 else 'PLAYER' if result < 0 else 'HOÀ'}")
        
        # In kết quả sau mỗi 5 trò chơi
        if (game_idx + 1) % 5 == 0:
            print(f"Thống kê sau {game_idx + 1} trò chơi:")
            print(f"AI thắng: {win_stats['AI']} ({win_stats['AI']/(game_idx+1)*100:.1f}%)")
            print(f"PLAYER thắng: {win_stats['PLAYER']} ({win_stats['PLAYER']/(game_idx+1)*100:.1f}%)")
            print(f"Hoà: {win_stats['DRAW']} ({win_stats['DRAW']/(game_idx+1)*100:.1f}%)")
    
    print(f"Thống kê cuối cùng - AI thắng: {win_stats['AI']}, PLAYER thắng: {win_stats['PLAYER']}, Hoà: {win_stats['DRAW']}")
    return training_data

# Cell 5: Hàm huấn luyện tối ưu
def train_model_cycle(cycles=3, games_per_cycle=20, epochs_per_cycle=10, mcts_simulations=500):
    """Huấn luyện mô hình theo chu trình lặp lại (tối ưu)"""
    # Tạo neural network mới hoặc tải model đã có
    model_path = 'models/connect4_model.h5' if os.path.exists('models/connect4_model.h5') else None
    nn = Connect4NeuralNetwork(model_path)
    agent = Connect4Agent(nn)
    
    training_history = []
    
    for cycle in range(cycles):
        cycle_start_time = time.time()
        print(f"=== Bắt đầu chu kỳ {cycle + 1}/{cycles} ===")
        
        # Thu thập dữ liệu từ tự chơi
        print("Thu thập dữ liệu tự chơi...")
        training_data = collect_self_play_data(agent, num_games=games_per_cycle, mcts_simulations=mcts_simulations)
        
        # Chuẩn bị dữ liệu huấn luyện
        boards, policies, values = nn.generate_training_data(training_data)
        
        print(f"Dữ liệu thu thập: {len(boards)} vị trí")
        
        # Huấn luyện mô hình
        print("Huấn luyện neural network...")
        history = nn.train(boards, policies, values, epochs=epochs_per_cycle, batch_size=32)
        training_history.append(history)
        
        # Lưu mô hình
        nn.save_model(f'models/connect4_model_cycle_{cycle + 1}.h5')
        nn.save_model('models/connect4_model.h5')  # Lưu mô hình mới nhất
        
        # Hiển thị kết quả huấn luyện
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['policy_accuracy'])
        plt.plot(history.history['val_policy_accuracy'])
        plt.title('Policy Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['value_mean_absolute_error'])
        plt.plot(history.history['val_value_mean_absolute_error'])
        plt.title('Value MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'models/training_cycle_{cycle + 1}.png')
        plt.show()
        
        # Đo thời gian chu kỳ
        cycle_time = time.time() - cycle_start_time
        print(f"Chu kỳ {cycle + 1} hoàn thành trong {cycle_time/60:.2f} phút")
    
    return nn, training_history

# Cell 6: Hàm đánh giá hiệu suất tối ưu
def evaluate_model(agent, num_games=20, opponent='mcts'):
    """Đánh giá hiệu suất của mô hình với số trò chơi nhỏ hơn"""
    win_stats = {'Agent': 0, 'Opponent': 0, 'Draw': 0}
    
    for game_idx in tqdm(range(num_games), desc="Evaluation games"):
        board = create_board()
        
        # Luân phiên người đi trước
        agent_first = game_idx % 2 == 0
        current_player = AI if agent_first else PLAYER
        
        while True:
            valid_moves = get_valid_locations(board)
            
            if not valid_moves:  # Bàn cờ đầy, hòa
                win_stats['Draw'] += 1
                break
                
            # Thực hiện nước đi
            if (current_player == AI and agent_first) or (current_player == PLAYER and not agent_first):
                # Agent's turn
                move, _ = agent.fast_mcts(board, valid_moves, 500)
            else:
                # Opponent's turn
                if opponent == 'mcts':
                    # MCTS thuần túy với số mô phỏng giảm
                    move = mcts_search(board, current_player, simulations=2000)
                else:
                    # Random opponent
                    move = random.choice(valid_moves)
            
            # Thực hiện nước đi
            row = get_next_open_row(board, move)
            drop_piece(board, row, move, current_player)
            
            # Kiểm tra thắng/thua
            if winning_move(board, current_player)[0]:
                if (current_player == AI and agent_first) or (current_player == PLAYER and not agent_first):
                    win_stats['Agent'] += 1
                else:
                    win_stats['Opponent'] += 1
                break
                
            # Chuyển lượt
            current_player = PLAYER if current_player == AI else AI
    
    print(f"Kết quả đánh giá với {opponent}:")
    print(f"Agent thắng: {win_stats['Agent']} ({win_stats['Agent']/num_games*100:.1f}%)")
    print(f"Opponent thắng: {win_stats['Opponent']} ({win_stats['Opponent']/num_games*100:.1f}%)")
    print(f"Hoà: {win_stats['Draw']} ({win_stats['Draw']/num_games*100:.1f}%)")
    
    return win_stats

# Cell 7: Chạy huấn luyện và đánh giá
# Khởi tạo mô hình và huấn luyện
nn, history = train_model_cycle(cycles=3, games_per_cycle=20, epochs_per_cycle=10, mcts_simulations=500)

# Tạo agent sử dụng mô hình đã huấn luyện
agent = Connect4Agent(nn)

# Đánh giá hiệu suất của agent so với MCTS thuần túy 
results_vs_mcts = evaluate_model(agent, num_games=20, opponent='mcts')

# Đánh giá hiệu suất của agent so với đối thủ ngẫu nhiên
results_vs_random = evaluate_model(agent, num_games=20, opponent='random')

# Cell 8: Lưu mô hình và kết quả
import pickle

# Lưu kết quả đánh giá
with open('models/evaluation_results.pkl', 'wb') as f:
    pickle.dump({'vs_mcts': results_vs_mcts, 'vs_random': results_vs_random}, f)

# Lưu lịch sử huấn luyện
with open('models/training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

print("Quá trình huấn luyện hoàn thành và đã lưu kết quả!")

# Cell 9: Tích hợp với game.py
def neural_mcts_move(board):
    """
    Hàm này có thể được sử dụng trong game.py để tích hợp mô hình đã huấn luyện
    """
    # Tải mô hình nếu cần
    model_path = 'models/connect4_model.h5'
    
    # Kiểm tra nếu có mô hình
    if os.path.exists(model_path):
        # Tạo agent với neural network
        nn = Connect4NeuralNetwork(model_path)
        agent = Connect4Agent(nn)
        
        # Lấy nước đi hợp lệ
        valid_moves = get_valid_locations(board)
        
        # Kết hợp neural network với MCTS
        # Chỉ sử dụng 500 mô phỏng thay vì 20,000
        move, _ = agent.fast_mcts(board, valid_moves, num_simulations=500)
        return move
    else:
        # Nếu không có mô hình, sử dụng MCTS thuần túy
        return mcts_search(board)

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo bàn cờ mới
    board = create_board()
    
    # Thực hiện một số nước đi thử nghiệm
    drop_piece(board, 5, 3, PLAYER)
    drop_piece(board, 5, 2, AI)
    drop_piece(board, 4, 3, PLAYER)
    
    # Sử dụng neural_mcts_move
    move = neural_mcts_move(board)
    print(f"Neural MCTS chọn cột: {move}")