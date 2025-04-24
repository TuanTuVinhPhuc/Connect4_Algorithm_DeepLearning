from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from settings import *
from ai import minimax
from MCTS import mcts_search
import numpy as np
import math

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
        
        max_player = game_state.current_player
        min_player = 2 if game_state.current_player == 1 else 1
        selected_move, _ = minimax(game_state.board, 7, -math.inf, math.inf, max_player, min_player, True, 3)
        print("🔍 GameState:", game_state.board)
        print("🎯 Valid Moves:", game_state.valid_moves)
        print("🧠 Current Player:", game_state.current_player)
        print("💻 Selected Move:", selected_move)
        return AIResponse(move=selected_move)
    except Exception as e:
        print("❌ Có lỗi xảy ra trong thuật toán AI:", e)
        import traceback
        traceback.print_exc()  # in ra stack trace đầy đủ

        if game_state.valid_moves:
            print("⚠️ Trả fallback move:", game_state.valid_moves[0])
            return AIResponse(move=int(game_state.valid_moves[0]))
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)