import numpy as np
import time as tm
import random as rand
import pygame as pg 

# Thiết lập màn hình
WIDTH = 800
rows = 6
columns = 7
square = WIDTH // columns
HEIGHT = (rows) * square
radius = (square // 2) - 5

# Khởi tạo Pygame
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Connect four gaming")

# Màu sắc
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

def draw_board():
    screen.fill(BLUE)
    for r in range(rows):
        for c in range(columns):
            pg.draw.circle(screen, WHITE, (c * square + square // 2, r * square + square // 2 + square), radius)

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    draw_board()
    pg.display.flip()

# Thoát game
pg.quit()