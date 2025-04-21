from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from settings import *
from ai import minimax
import math
from MCTS import mcts_search
import numpy as np
from ai_none_tatic import minimax_v1
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        
        # change logic thuật toán AI của bạn ở đây
        print("Current player:", game_state.current_player)
        print("board in global")
        for row in game_state.board:
           print(row)

        board = np.array(game_state.board)
        current_player = game_state.current_player
  
        start_time = time.time()

        if current_player == 1:
            # flipped_board = board.copy()
            # flipped_board[flipped_board == 1] = -1  # Giá trị tạm thời
            # flipped_board[flipped_board == 2] = 1
            # flipped_board[flipped_board == -1] = 2 

            # selected_move, _ = minimax(flipped_board, 10, -math.inf, math.inf, AI, True)  # vì AI =2 mà ta lại đang đánh quân 1

            selected_move, _ = mcts_search(board, PLAYER, neural_network=None, simulations=50000)
        elif current_player == 2:
            selected_move, _ = minimax(board, 10, -math.inf, math.inf, AI, True) 

        end_time = time.time()
        print(f"Minimax - thời gian suy nghĩ: {end_time - start_time:.3f}s")
        print("minimax đang chơi với số thứ tự: ", current_player)

        return AIResponse(move=selected_move)
    except Exception as e:
        print("❌ Có lỗi xảy ra trong thuật toán AI:", e)
        import traceback
        traceback.print_exc()  # in ra stack trace đầy đủ

        if game_state.valid_moves:
            print("⚠️ Trả fallback move:", game_state.valid_moves[0])
            return AIResponse(move=int(game_state.valid_moves[0]))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)