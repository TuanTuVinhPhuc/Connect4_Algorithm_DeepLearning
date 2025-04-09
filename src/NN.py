import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, BatchNormalization, Activation, add, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import importlib

from tqdm.notebook import tqdm

from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY
import MCTS
importlib.reload(MCTS)
from MCTS import mcts_search

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

class Connect4NeuralNetwork:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self._prediction_cache = {}
        self.model_ready = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
            
    def build_model(self):
        input_shape = (ROWS, COLUMNS, 3)
        inputs = Input(shape=input_shape)
        
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
   
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 64)
        
        policy_head = Conv2D(32, (1, 1), padding='same', kernel_regularizer=l2(0.0001))(x)
        policy_head = BatchNormalization()(policy_head)
        policy_head = Activation('relu')(policy_head)
        policy_head = Flatten()(policy_head)
        policy_head = Dropout(0.1)(policy_head)
        policy_head = Dense(COLUMNS, activation='softmax', name='policy', kernel_regularizer=l2(0.0001))(policy_head)
        
        value_head = Conv2D(32, (1, 1), padding='same', kernel_regularizer=l2(0.0001))(x)
        value_head = BatchNormalization()(value_head)
        value_head = Activation('relu')(value_head)
        value_head = Flatten()(value_head)
        value_head = Dense(64, activation='relu', kernel_regularizer=l2(0.00015))(value_head)
        value_head = Dropout(0.5)(value_head)
        value_head = Dense(1, activation='tanh', name='value', kernel_regularizer=l2(0.00015))(value_head)
        
        self.model = Model(inputs=inputs, outputs=[policy_head, value_head])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.00008),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            },
            loss_weights={
                'policy': 1.0,
                'value': 2.0  # Tăng trọng số cho value
            },
            metrics={
                'policy': 'accuracy',
                'value': 'mean_absolute_error'
            }
        )
        
        self.model.summary()
        self.model_ready = True
    
    # Học phần chênh lệch
    def _residual_block(self, x, filters):
        shortcut = x
        
        # x = Conv2D(filters, (3, 3), padding='same')(x)
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # x = Conv2D(filters, (3, 3), padding='same')(x)
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
        x = BatchNormalization()(x)
        
        x = add([x, shortcut])
        x = Activation('relu')(x)
        
        return x
    
    def _board_to_key(self, board):
        key = []
        for r in range(ROWS):
            for c in range(COLUMNS):
                key.append(str(board[r][c]))
        return "".join(key)
    
    def _board_to_tensor(self, board):
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
        if not self.model_ready:
            default_policy = np.ones(COLUMNS) / COLUMNS
            default_value = 0.0
            return default_policy, default_value
        
        board_key = self._board_to_key(board)
        
        if board_key in self._prediction_cache:
            return self._prediction_cache[board_key]
        
        board_tensor = self._board_to_tensor(board)
        policy, value = self.model.predict(np.expand_dims(board_tensor, axis=0), verbose=0)

        result = (policy[0], value[0][0])
        
        self._prediction_cache[board_key] = result
        
        return result
    
    def clear_cache(self):
        self._prediction_cache = {}
    
    def train(self, boards, policies, values, epochs=15, batch_size=64, validation_split=0.2):
        X = np.array([self._board_to_tensor(board) for board in boards])
        y_policy = np.array(policies, dtype=np.float32)
        y_value = np.array(values, dtype=np.float32).reshape(-1, 1)
        
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
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.00001
            )
        ]
        
        history = self.model.fit(
            X, 
            {'policy': y_policy, 'value': y_value},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.clear_cache()
        
        return history
    
    def save_model(self, path='models/connect4_model.h5'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model đã được lưu tại {path}")
    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print(f"Model đã được tải từ {path}")
        self.model_ready = True

        self.model.compile(
        optimizer=Adam(learning_rate=0.00008), 
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mean_squared_error'
        },
        loss_weights={
        'policy': 1.0,
        'value': 2.0  # Tăng trọng số cho value
        },
        metrics={
            'policy': 'accuracy',
            'value': 'mean_absolute_error'
        }
    )
    
    def generate_training_data(self, mcts_games, augment=True):
        boards = []
        policies = []
        values = []
        
        for board, policy, result in mcts_games:
            boards.append(board)
            policies.append(policy)
            values.append(result)
  
            if augment:
                flipped_board = np.flip(board, axis=1).copy()
                flipped_policy = np.flip(policy).copy()
                boards.append(flipped_board)
                policies.append(flipped_policy)
                values.append(result)
        
        return boards, policies, values
    
class Connect4Agent:
    def __init__(self, neural_network=None, model_path=None):

        if neural_network:
            self.nn = neural_network
        else:
            self.nn = Connect4NeuralNetwork(model_path)
        
        self.temperature = 1.0  
        self.batch_size = 32  
    
    def fast_mcts(self, board, num_simulations=2000, temperature_decay=True):
        pieces_count = np.count_nonzero(board != EMPTY)
        total_positions = ROWS * COLUMNS
        game_progress = pieces_count / total_positions  # 0.0 đến 1.0

        if temperature_decay:
            temperature = max(0.3, 1.0 - game_progress * 0.7)
        else:
            temperature = 1.0

        move, mcts_policy = mcts_search(board, AI, self.nn, num_simulations)
        
        if temperature != 1.0:
            mcts_policy = np.power(mcts_policy + 1e-10, 1.0/temperature)
            mcts_policy /= np.sum(mcts_policy)
            
        return move, mcts_policy

