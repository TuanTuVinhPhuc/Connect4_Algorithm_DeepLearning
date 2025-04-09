# MCTS
import pygame as pg
import sys
from time import sleep
import random as rd

from settings import *
from board import create_board, drop_piece, is_valid_location, get_next_open_row, winning_move, get_valid_locations
from MCTS import random_move as mcts_move
from utils import create_gradient_background, draw_board, animate_piece_drop, show_game_over

def main():
    board = create_board()
    game_over = False
    turn = rd.randint(0,1)  # Player 1 starts (0 = Player, 1 = AI)
    
    background = create_gradient_background()
    
    while True:
        highlighted_col = None
        winner_cells = None
        
        # Xử lý sự kiện
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
                
            if event.type == pg.MOUSEMOTION and not game_over:

                mouse_x = event.pos[0]
                highlighted_col = int(mouse_x // SQUARE_SIZE)
                
            if event.type == pg.MOUSEBUTTONDOWN and not game_over:
        
                if turn == 0:
                    mouse_x = event.pos[0]
                    col = int(mouse_x // SQUARE_SIZE)
                    
                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        animate_piece_drop(screen, board, row, col, PLAYER, background, ANIMATION_SPEED)
                        drop_piece(board, row, col, PLAYER)
                        
                        win, winning_cells = winning_move(board, PLAYER)
                        if win:
                            winner_cells = winning_cells
                            draw_board(screen, board, background, None, None, winner_cells)
                            show_game_over(screen, "PLAYER đã chiến thắng!", background)
                            game_over = True
                            continue
                            
                        if len(get_valid_locations(board)) == 0:
                            draw_board(screen, board, background)
                            show_game_over(screen, "Hòa!", background)
                            game_over = True
                            continue
                            
                        turn = 1  # Switch to AI
        
        # AI's turn
        if turn == 1 and not game_over:
            sleep(0.5)
            
            col, policy = mcts_move(board)
            
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                animate_piece_drop(screen, board, row, col, AI, background, ANIMATION_SPEED)
                drop_piece(board, row, col, AI)
                
                # Kiểm tra AI thắng
                win, winning_cells = winning_move(board, AI)
                if win:
                    winner_cells = winning_cells
                    draw_board(screen, board, background, None, None, winner_cells)
                    show_game_over(screen, "AI đã chiến thắng! Tuấn gà", background)
                    game_over = True
                    continue
                    
                # Kiểm tra hòa
                if len(get_valid_locations(board)) == 0:
                    draw_board(screen, board, background)
                    show_game_over(screen, "Hòa!", background)
                    game_over = True
                    continue
                    
                turn = 0  # Switch to player
                
        draw_board(screen, board, background, highlighted_col, None, winner_cells)

if __name__ == "__main__":
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Connect Four Game")
    main()



# Minimax
# import pygame as pg
# import sys
# from time import sleep
# import math

# from settings import *
# from board import create_board, drop_piece, is_valid_location, get_next_open_row, winning_move, get_valid_locations
# from ai import minimax
# from utils import create_gradient_background, draw_board, animate_piece_drop, show_game_over

# def main():
#     # Tạo bảng chơi
#     board = create_board()
#     game_over = False
#     turn = 1  # Player 1 starts (0 = Player, 1 = AI)
    
#     # Tạo background
#     background = create_gradient_background()
    
#     # Vòng lặp chính
#     while True:
#         highlighted_col = None
#         winner_cells = None
        
#         # Xử lý sự kiện
#         for event in pg.event.get():
#             if event.type == pg.QUIT:
#                 pg.quit()
#                 sys.exit()
                
#             if event.type == pg.MOUSEMOTION and not game_over:
#                 # Highlight cột khi di chuột
#                 mouse_x = event.pos[0]
#                 highlighted_col = int(mouse_x // SQUARE_SIZE)
                
#             if event.type == pg.MOUSEBUTTONDOWN and not game_over:
#                 # Player's turn
#                 if turn == 0:
#                     mouse_x = event.pos[0]
#                     col = int(mouse_x // SQUARE_SIZE)
                    
#                     if is_valid_location(board, col):
#                         row = get_next_open_row(board, col)
#                         animate_piece_drop(screen, board, row, col, PLAYER, background, ANIMATION_SPEED)
#                         drop_piece(board, row, col, PLAYER)
                        
#                         # Kiểm tra người chơi thắng
#                         win, winning_cells = winning_move(board, PLAYER)
#                         if win:
#                             winner_cells = winning_cells
#                             draw_board(screen, board, background, None, None, winner_cells)
#                             show_game_over(screen, "Bạn đã chiến thắng!", background)
#                             game_over = True
#                             board = create_board()
#                             game_over = False
#                             turn = 0
#                             continue
                            
#                         # Kiểm tra hòa
#                         if len(get_valid_locations(board)) == 0:
#                             draw_board(screen, board, background)
#                             show_game_over(screen, "Hòa!", background)
#                             game_over = True
#                             board = create_board()
#                             game_over = False
#                             turn = 0
#                             continue
                            
#                         turn = 1  # Switch to AI
        
#         # AI's turn
#         if turn == 1 and not game_over:
#             # Tạo hiệu ứng suy nghĩ
#             sleep(0.5)
            
#             # AI chọn cột - độ sâu AI_DIFFICULTY là đủ tốt và không quá chậm
#             col, minimax_score = minimax(board, AI_DIFFICULTY, -math.inf, math.inf, True)
            
#             if is_valid_location(board, col):
#                 row = get_next_open_row(board, col)
#                 animate_piece_drop(screen, board, row, col, AI, background, ANIMATION_SPEED)
#                 drop_piece(board, row, col, AI)
                
#                 # Kiểm tra AI thắng
#                 win, winning_cells = winning_move(board, AI)
#                 if win:
#                     winner_cells = winning_cells
#                     draw_board(screen, board, background, None, None, winner_cells)
#                     show_game_over(screen, "AI đã chiến thắng!", background)
#                     game_over = True
#                     board = create_board()
#                     game_over = False
#                     turn = 0
#                     continue
                    
#                 # Kiểm tra hòa
#                 if len(get_valid_locations(board)) == 0:
#                     draw_board(screen, board, background)
#                     show_game_over(screen, "Hòa!", background)
#                     game_over = True
#                     board = create_board()
#                     game_over = False
#                     turn = 0
#                     continue
                    
#                 turn = 0  # Switch to player
                
#         # Vẽ bảng
#         draw_board(screen, board, background, highlighted_col, None, winner_cells)

# if __name__ == "__main__":
#     # Khởi tạo màn hình trong game.py để tránh trùng lặp
#     screen = pg.display.set_mode((WIDTH, HEIGHT))
#     pg.display.set_caption("Connect Four Game")
#     main()