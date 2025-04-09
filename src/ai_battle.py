import os
import sys
import pygame as pg
import numpy as np
import time
import math
from time import sleep
import random as rd

from settings import ROWS, COLUMNS, SQUARE_SIZE, PLAYER, AI, EMPTY, AI_DIFFICULTY
from board import create_board, drop_piece, is_valid_location, get_next_open_row, winning_move, get_valid_locations
from utils import create_gradient_background, draw_board, animate_piece_drop, show_game_over
from MCTS import mcts_search
from ai import minimax
from NN import Connect4NeuralNetwork, Connect4Agent 

minimax_wins = 0
mcts_wins = 0
draws = 0
games_played = 0
total_minimax_time = 0
total_mcts_time = 0

def draw_thinking_status(screen, player, dots, calculation_time):
    # Tính toán hiệu ứng animation
    dots_str = "." * (dots // 10)
    
    # Tạo gradient cho nền
    overlay_width = 450
    overlay_height = 60
    overlay = pg.Surface((overlay_width, overlay_height), pg.SRCALPHA)
    
    # Gradient từ trong suốt đến đen mờ
    for y in range(overlay_height):
        alpha = min(180, int(y * 3 + 60))  # Độ mờ tăng dần từ trên xuống
        pg.draw.line(overlay, (0, 0, 0, alpha), (0, y), (overlay_width, y))
    
    # Vẽ viền bo tròn mờ cho overlay
    pg.draw.rect(overlay, (255, 255, 255, 30), overlay.get_rect(), 1, border_radius=10)
    
    # Hiển thị overlay
    overlay_x = SQUARE_SIZE * COLUMNS // 2 - overlay_width // 2
    overlay_y = 10
    screen.blit(overlay, (overlay_x, overlay_y))
    
    # Màu sắc tùy thuộc AI nào đang suy nghĩ
    if player == 0:  # Minimax AI (AI1)
        color = (100, 200, 255)  # Xanh dương
        name = "Minimax AI"
    else:  # MCTS AI (AI2)
        color = (255, 180, 100)  # Cam
        name = "MCTS AI"
    
    # Vẽ text chính
    font = pg.font.SysFont('Arial', 22)
    text = font.render(f"{name} đang suy nghĩ{dots_str}", True, color)
    text_rect = text.get_rect(center=(SQUARE_SIZE * COLUMNS // 2, overlay_y + 25))
    screen.blit(text, text_rect)
    
    # Vẽ thời gian tính toán
    time_color = color
    
    # Hiển thị thời gian với hiệu ứng thanh tiến trình
    progress_width = (overlay_width - 40) * min(1.0, calculation_time / 2.0)
    progress_rect = pg.Rect(overlay_x + 20, overlay_y + 42, progress_width, 6)
    
    # Vẽ nền thanh tiến trình
    pg.draw.rect(screen, (60, 60, 60, 180), 
                 (overlay_x + 20, overlay_y + 42, overlay_width - 40, 6), 
                 border_radius=3)
    
    # Vẽ thanh tiến trình
    if progress_width > 0:
        pg.draw.rect(screen, time_color, progress_rect, border_radius=3)
    
    # Hiển thị thời gian
    time_font = pg.font.SysFont('Arial', 16)
    time_text = time_font.render(f"{calculation_time:.3f}s", True, time_color)
    time_rect = time_text.get_rect(midright=(overlay_x + overlay_width - 20, overlay_y + 45))
    screen.blit(time_text, time_rect)
    
    pg.display.update()

def update_stats(screen, background):
    """Hiển thị thống kê trận đấu"""
    global games_played, total_minimax_time, total_mcts_time
    
    avg_minimax_time = total_minimax_time / max(games_played, 1)
    avg_mcts_time = total_mcts_time / max(games_played, 1)
    
    stats_text = f"Minimax: {minimax_wins} ({avg_minimax_time:.3f}s)  |  MCTS: {mcts_wins} ({avg_mcts_time:.3f}s)  |  Hòa: {draws}"
    games_text = f"Trận đã chơi: {games_played}"
    
    # Tạo overlay
    overlay = pg.Surface((SQUARE_SIZE * COLUMNS, 50))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Vẽ text
    font = pg.font.SysFont('Arial', 20)
    text_surface = font.render(stats_text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(SQUARE_SIZE * COLUMNS // 2, 20))
    screen.blit(text_surface, text_rect)
    
    games_surface = font.render(games_text, True, (200, 200, 200))
    games_rect = games_surface.get_rect(center=(SQUARE_SIZE * COLUMNS // 2, 40))
    screen.blit(games_surface, games_rect)
    
    # Hiển thị hướng dẫn
    help_text = "SPACE: Reset  |  A: Auto play  |  R: Random mở đầu  |  ESC: Thoát"
    help_surface = font.render(help_text, True, (200, 200, 200))
    help_rect = help_surface.get_rect(center=(SQUARE_SIZE * COLUMNS // 2, ROWS * SQUARE_SIZE + SQUARE_SIZE / 2))
    screen.blit(help_surface, help_rect)
    
    pg.display.update()

def main(agent):
    global minimax_wins, mcts_wins, draws, games_played, total_minimax_time, total_mcts_time

    # mm = Connect4NeuralNetwork('models/connect4_model_cycle_2.h5')
    # agent1 = Connect4Agent(mm)

    # Khởi tạo pygame
    pg.init()
    pg.font.init()
    screen = pg.display.set_mode((SQUARE_SIZE * COLUMNS, (ROWS + 1) * SQUARE_SIZE))
    pg.display.set_caption("AI Battle: Minimax vs MCTS")
    
    # Tạo bảng chơi
    board = create_board()
    game_over = False
    turn = rd.randint(0, 1)  # Chọn ngẫu nhiên AI nào đi trước
    pause_time = 0.3  # Thời gian chờ giữa các nước đi
    
    # Tạo background
    background = create_gradient_background()
    
    # Vòng lặp chính
    clock = pg.time.Clock()
    fps = 60
    auto_play = False
    auto_restart_delay = 1  # seconds
    restart_counter = 0
    
    # Biến hiệu ứng
    thinking_dots = 0
    ai_thinking = False
    thinking_start = 0
    
    # Vòng lặp chính
    while True:
        clock.tick(fps)
        winner_cells = None
        thinking_dots = (thinking_dots + 1) % 40
        
        # Xử lý sự kiện
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    # Reset game nếu nhấn SPACE
                    board = create_board()
                    game_over = False
                    turn = rd.randint(0, 1)
                    draw_board(screen, board, background)
                    update_stats(screen, background)
                    ai_thinking = False
                    continue
                elif event.key == pg.K_a:
                    # Bật/tắt auto play
                    auto_play = not auto_play
                    print(f"Auto play: {'Bật' if auto_play else 'Tắt'}")
                    continue
                elif event.key == pg.K_r:
                    # Chọn ngẫu nhiên AI nào đi trước
                    board = create_board()
                    game_over = False
                    turn = rd.randint(0, 1)
                    draw_board(screen, board, background)
                    update_stats(screen, background)
                    ai_thinking = False
                    continue
                elif event.key == pg.K_ESCAPE:
                    pg.quit()
                    sys.exit()
        
        # Auto restart sau khi game kết thúc nếu bật auto play
        if game_over and auto_play:
            restart_counter += 1/fps
            if restart_counter >= auto_restart_delay:
                board = create_board()
                game_over = False
                turn = rd.randint(0, 1)
                restart_counter = 0
                draw_board(screen, board, background)
                update_stats(screen, background)
        
        # AI1's turn (Minimax)
        if turn == 0 and not game_over:
            if not ai_thinking:
                # Bắt đầu suy nghĩ
                ai_thinking = True
                thinking_start = time.time()
            
            # Hiển thị trạng thái đang suy nghĩ
            calc_time = time.time() - thinking_start
            draw_thinking_status(screen, 0, thinking_dots, calc_time)
            
            if calc_time >= pause_time:
                thinking_start = time.time()
                col, _ = agent.fast_mcts(board, 300, temperature_decay=True)
                # col, _ = mcts_search(board, AI, neural_network=None, simulations=20000)
                thinking_time = time.time() - thinking_start
                total_minimax_time += thinking_time
                
                if col is not None and is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    animate_piece_drop(screen, board, row, col, PLAYER, background, 15)
                    drop_piece(board, row, col, PLAYER)
                    
                    print(f"NN đã chọn cột {col} (thời gian: {thinking_time:.3f}s)")
                    
                    # Vẽ bảng sau khi đi
                    draw_board(screen, board, background)
                    update_stats(screen, background)
                    
                    # Kiểm tra AI1 thắng
                    win, winning_cells = winning_move(board, PLAYER)
                    if win:
                        winner_cells = winning_cells
                        draw_board(screen, board, background, None, None, winner_cells)
                        show_game_over(screen, "Neraul network AI thắng!", background)
                        minimax_wins += 1
                        games_played += 1
                        update_stats(screen, background)
                        game_over = True
                        restart_counter = 0
                        continue
                        
                    # Kiểm tra hòa
                    if len(get_valid_locations(board)) == 0:
                        draw_board(screen, board, background)
                        show_game_over(screen, "Hòa!", background)
                        draws += 1
                        games_played += 1
                        update_stats(screen, background)
                        game_over = True
                        restart_counter = 0
                        continue
                        
                    turn = 1  # Chuyển sang MCTS AI
                    ai_thinking = False
        
        # AI2's turn (MCTS)
        if turn == 1 and not game_over:
            if not ai_thinking:
                # Bắt đầu suy nghĩ
                ai_thinking = True
                thinking_start = time.time()
            
            # Hiển thị trạng thái đang suy nghĩ
            calc_time = time.time() - thinking_start
            draw_thinking_status(screen, 1, thinking_dots, calc_time)
            
            # Sử dụng AI MCTS
            if calc_time >= pause_time:
                thinking_start = time.time()
                # col, _ = agent1.fast_mcts(board, 300, temperature_decay=False)
                col, _ = mcts_search(board, AI, neural_network=None, simulations=10000)
                # col, _ = minimax(board, AI_DIFFICULTY, -math.inf, math.inf, True)
                thinking_time = time.time() - thinking_start
                total_mcts_time += thinking_time
                
                if col is not None and is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    animate_piece_drop(screen, board, row, col, AI, background, 15)
                    drop_piece(board, row, col, AI)
                    
                    print(f"MCTS đã chọn cột {col} (thời gian: {thinking_time:.3f}s)")
                    
                    # Vẽ bảng sau khi đi
                    draw_board(screen, board, background)
                    update_stats(screen, background)
                    
                    # Kiểm tra AI2 thắng
                    win, winning_cells = winning_move(board, AI)
                    if win:
                        winner_cells = winning_cells
                        draw_board(screen, board, background, None, None, winner_cells)
                        show_game_over(screen, "MCTS AI thắng!", background)
                        mcts_wins += 1
                        games_played += 1
                        update_stats(screen, background)
                        game_over = True
                        restart_counter = 0
                        continue
                        
                    # Kiểm tra hòa
                    if len(get_valid_locations(board)) == 0:
                        draw_board(screen, board, background)
                        show_game_over(screen, "Hòa!", background)
                        draws += 1
                        games_played += 1
                        update_stats(screen, background)
                        game_over = True
                        restart_counter = 0
                        continue
                        
                    turn = 0  # Chuyển sang NN
                    ai_thinking = False
                
        # Vẽ bảng
        if not ai_thinking:
            draw_board(screen, board, background, None, None, winner_cells)
            update_stats(screen, background)

if __name__ == "__main__":
    # Hiển thị thông tin khi khởi động
    print("=== AI Battle: Minimax vs MCTS ===")
    print("Phím tắt:")
    print("  SPACE: Reset game")
    print("  A: Bật/tắt tự động chơi")
    print("  R: Chọn ngẫu nhiên AI đi trước")
    print("  ESC: Thoát")
    print("=====================================")
    
    # Chạy game
    nn = Connect4NeuralNetwork('models/connect4_model.h5')
    agent = Connect4Agent(nn)
    main(agent)


