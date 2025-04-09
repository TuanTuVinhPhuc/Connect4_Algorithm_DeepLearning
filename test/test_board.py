import os
import sys
import numpy as np
import pygame as pg
import math
from time import sleep

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from settings import ROWS, COLUMNS, PLAYER, AI, EMPTY, AI_DIFFICULTY, SQUARE_SIZE
from board import drop_piece, is_valid_location, get_next_open_row, winning_move, get_valid_locations
from utils import create_gradient_background, draw_board, animate_piece_drop, show_game_over
from ai import minimax

def create_test_board():
    board = np.zeros((ROWS, COLUMNS))

    #TH1
    board[5, 1] = PLAYER
    board[5, 2] = PLAYER
    board[5, 3] = PLAYER
    board[5, 6] = PLAYER
    board[4, 6] = PLAYER
    board[4, 5] = PLAYER
    board[5, 5] = AI
    board[4, 3] = AI
    board[4, 4] = AI
    board[5, 0] = AI
    board[4, 1] = AI
    board[5, 4] = AI
    board[3,6] = PLAYER

    return board

def main():
    pg.init()
    pg.font.init()
    screen = pg.display.set_mode((800, (ROWS + 1) * SQUARE_SIZE))
    pg.display.set_caption("Test AI Decision")
    
    # Tạo bàn cờ test
    board = create_test_board()
    background = create_gradient_background()
    
    # Vẽ bàn cờ
    draw_board(screen, board, background)
    
    print("Nhấn ENTER để xem AI chọn nước đi tiếp theo.")
    print("Nhấn SPACE để tạo bàn cờ mới.")
    print("Nhấn ESC để thoát.")
    
    waiting = True
    while waiting:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
                
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:  
                    print("AI đang suy nghĩ...")
                    col, minimax_score = minimax(board, AI_DIFFICULTY, -math.inf, math.inf, True)
                    
                    if col is not None and is_valid_location(board, col):
                        print(f"AI chọn cột: {col} với điểm số: {minimax_score}")
                        
                        row = get_next_open_row(board, col)
                        animate_piece_drop(screen, board, row, col, AI, background, 15)
                        drop_piece(board, row, col, AI)
                        
                        # Vẽ lại bàn cờ
                        draw_board(screen, board, background)
                        
                        # Kiểm tra AI thắng
                        win, winning_cells = winning_move(board, AI)
                        if win:
                            print("AI đã thắng!")
                            draw_board(screen, board, background, None, None, winning_cells)
                            
                elif event.key == pg.K_SPACE:  # SPACE
                    # Tạo bàn cờ mới
                    board = create_test_board()
                    draw_board(screen, board, background)
                    
                elif event.key == pg.K_ESCAPE:  # ESC
                    pg.quit()
                    sys.exit()

if __name__ == "__main__":
    main()

