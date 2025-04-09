import pygame as pg
import numpy as np
import sys
from settings import WIDTH, HEIGHT, SQUARE_SIZE, RADIUS, FONT, PLAYER, AI
from settings import BLACK, WHITE, RED, YELLOW, BLUE, DARK_BLUE, LIGHT_BLUE

def create_gradient_background():
    background = pg.Surface((WIDTH, HEIGHT))
    for y in range(HEIGHT):
        # Calculate color gradient from top to bottom
        ratio = y / HEIGHT
        r = int(50 + ratio * 10)
        g = int(50 + ratio * 10)
        b = int(200 - ratio * 50)
        pg.draw.line(background, (r, g, b), (0, y), (WIDTH, y))
    return background

def draw_board(screen, board, background, highlight_col=None, animation_piece=None, winner_cells=None):
    screen.blit(background, (0, 0))
    if highlight_col is not None:
        pg.draw.rect(screen, LIGHT_BLUE, (highlight_col * SQUARE_SIZE, 0, SQUARE_SIZE, SQUARE_SIZE))
    
    # Vẽ vị trí chưa quân cờ 
    for c in range(len(board[0])):
        for r in range(len(board)):
            pg.draw.rect(screen, BLUE, (c * SQUARE_SIZE, (r + 1) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pg.draw.circle(screen, BLACK, (int(c * SQUARE_SIZE + SQUARE_SIZE/2), int((r + 1) * SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS + 2)
            pg.draw.circle(screen, DARK_BLUE, (int(c * SQUARE_SIZE + SQUARE_SIZE/2), int((r + 1) * SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS)
    
    # Vẽ các quân cờ
    for c in range(len(board[0])):
        for r in range(len(board)):
            if board[r][c] == PLAYER:  # Player
                color = RED
                if winner_cells and (r, c) in winner_cells:
                    # Highlight khi đúp chuột ra ngoài khu vực chơi
                    pg.draw.circle(screen, WHITE, (int(c * SQUARE_SIZE + SQUARE_SIZE/2), int((r + 1) * SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS + 4)
                pg.draw.circle(screen, color, (int(c * SQUARE_SIZE + SQUARE_SIZE/2), int((r + 1) * SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS)
            elif board[r][c] == AI:  # AI
                color = YELLOW
                if winner_cells and (r, c) in winner_cells:
                    pg.draw.circle(screen, WHITE, (int(c * SQUARE_SIZE + SQUARE_SIZE/2), int((r + 1) * SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS + 4)
                pg.draw.circle(screen, color, (int(c * SQUARE_SIZE + SQUARE_SIZE/2), int((r + 1) * SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS)
    
    # Vẽ quân đang rơi 
    if animation_piece:
        piece, col, y_pos = animation_piece
        color = RED if piece == PLAYER else YELLOW
        pg.draw.circle(screen, color, (int(col * SQUARE_SIZE + SQUARE_SIZE/2), int(y_pos)), RADIUS)
    
    pg.display.update()

def animate_piece_drop(screen, board, row, col, piece, background, animation_speed):
    start_y = SQUARE_SIZE / 2
    end_y = (row + 1) * SQUARE_SIZE + SQUARE_SIZE/2
    
    # Vẽ quá trình rơi
    for y_pos in np.linspace(start_y, end_y, 20):
        draw_board(screen, board, background, None, (piece, col, y_pos))
        pg.time.wait(animation_speed)

def draw_text(screen, text, color, x, y):
    font = pg.font.SysFont('Arial', 36)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)

def show_game_over(screen, message, background):
    screen.blit(background, (0, 0))
    
    # Draw semi-transparent overlay
    overlay = pg.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    # Draw message
    draw_text(screen, message, WHITE, WIDTH//2, HEIGHT//2 - 50)
    
    # Nếu không phải đánh giá mô hình, thêm thông báo nhấn SPACE để chơi lại
    if "Agent" not in message:  # Dành cho chế độ chơi thường
        draw_text(screen, "Nhấn SPACE để chơi lại", WHITE, WIDTH//2, HEIGHT//2 + 30)
        
        pg.display.update()
        
        # Chờ người dùng nhấn phím
        waiting = True
        while waiting:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        waiting = False
                    elif event.key == pg.K_ESCAPE:
                        pg.quit()
                        sys.exit()
    else:  # Dành cho chế độ đánh giá mô hình
        pg.display.update()
        pg.time.wait(3000)  # Chờ 3 giây

def draw_thinking_status(screen, thinking_dots, calculation_time):
    # Tính toán hiệu ứng animation
    dots = "." * (thinking_dots // 10)
    
    # Tạo gradient cho nền
    overlay_width = 400
    overlay_height = 60
    overlay = pg.Surface((overlay_width, overlay_height), pg.SRCALPHA)
    
    # Gradient từ trong suốt đến đen mờ
    for y in range(overlay_height):
        alpha = min(180, int(y * 3 + 60))  # Độ mờ tăng dần từ trên xuống
        pg.draw.line(overlay, (0, 0, 0, alpha), (0, y), (overlay_width, y))
    
    # Vẽ viền bo tròn mờ cho overlay
    pg.draw.rect(overlay, (255, 255, 255, 30), overlay.get_rect(), 1, border_radius=10)
    
    # Hiển thị overlay
    overlay_x = screen.get_width() // 2 - overlay_width // 2
    overlay_y = 10
    screen.blit(overlay, (overlay_x, overlay_y))
    
    # Tạo hiệu ứng glow cho text
    glow_size = int(thinking_dots % 20) // 3 + 1
    
    # Vẽ "glow" mờ xung quanh text
    for i in range(1, glow_size + 1):
        glow_font = pg.font.SysFont('Arial', 21)
        glow_text = glow_font.render(f"Con mẹ chờ bố mày tí{dots}", True, (100, 100, 255))  # RGB
        glow_text.set_alpha(max(0, 150 - i * 30))  # Đặt độ trong suốt
        glow_rect = glow_text.get_rect(center=(screen.get_width() // 2, overlay_y + 25))
        offset = i
        screen.blit(glow_text, (glow_rect.x - offset, glow_rect.y))
        screen.blit(glow_text, (glow_rect.x + offset, glow_rect.y))
        screen.blit(glow_text, (glow_rect.x, glow_rect.y - offset))
        screen.blit(glow_text, (glow_rect.x, glow_rect.y + offset))
    
    # Vẽ text chính
    font = pg.font.SysFont('Arial', 22)
    text = font.render(f"Con mẹ chờ bố mày tí{dots}", True, (220, 220, 255))
    text_rect = text.get_rect(center=(screen.get_width() // 2, overlay_y + 25))
    screen.blit(text, text_rect)
    
    # Vẽ thời gian tính toán với màu khác
    time_color = (255, 220, 100)  # Màu vàng nhạt
    if calculation_time > 4.0:
        time_color = (255, 100, 100)  # Đỏ nếu gần hết thời gian
    elif calculation_time > 3.0:
        time_color = (255, 150, 50)  # Cam nếu trung bình
    
    # Hiển thị thanh tiến trình
    progress_width = (overlay_width - 40) * min(1.0, calculation_time / 5.0)
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
    time_text = time_font.render(f"{calculation_time:.1f}s / 5.0s", True, time_color)
    time_rect = time_text.get_rect(midright=(overlay_x + overlay_width - 20, overlay_y + 45))
    screen.blit(time_text, time_rect)
    
    pg.display.update()