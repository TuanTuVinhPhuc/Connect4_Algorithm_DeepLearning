import numpy as np
from settings import ROWS, COLUMNS, EMPTY, PLAYER, AI

def create_board():
    return np.zeros((ROWS, COLUMNS))

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[0][col] == EMPTY

def get_next_open_row(board, col):
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return -1

def winning_move(board, piece):
    # Kiểm tra hàng ngang
    for c in range(COLUMNS-3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True, [(r, c), (r, c+1), (r, c+2), (r, c+3)]

    # Kiểm tra hàng dọc
    for c in range(COLUMNS):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True, [(r, c), (r+1, c), (r+2, c), (r+3, c)]

    # Kiểm tra đường chéo lên
    for c in range(COLUMNS-3):
        for r in range(ROWS-3):
            if board[r+3][c] == piece and board[r+2][c+1] == piece and board[r+1][c+2] == piece and board[r][c+3] == piece:
                return True, [(r+3, c), (r+2, c+1), (r+1, c+2), (r, c+3)]

    # Kiểm tra đường chéo xuống
    for c in range(COLUMNS-3):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True, [(r, c), (r+1, c+1), (r+2, c+2), (r+3, c+3)]

    return False, []

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def is_terminal_node(board):
    return winning_move(board, PLAYER)[0] or winning_move(board, AI)[0] or len(get_valid_locations(board)) == 0