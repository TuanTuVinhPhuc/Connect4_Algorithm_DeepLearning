# 🎮 CompetitionAI

---

## 🧠 Introduction
This is a classic **Connect4** game where you can compete against various AI types: from **Minimax**, **Heuristic**, **MCTS**, to a **simple Neural Network**.

The project is divided into **2 main parts**:

- **Interface**: Manages the board, initializes, and displays the game (using **Pygame** or other graphics libraries).
- **AI**: Includes artificial intelligence algorithms to help the computer make optimal moves.

---

## 📁 Project Structure
<pre>
connect4/
├── src/
│   ├── setting.py         # Fixed board parameters (size, colors, etc.)
│   ├── board.py           # Logic for initializing and updating the board
│   ├── ultis.py           # Helper functions for board operations
│   ├── game.py            # Interface for human vs AI matches
│   ├── ai_battle.py       # AI vs AI matches to compare algorithm strengths
│   ├── MCTS.py            # Monte Carlo Tree Search algorithm
│   ├── ai.py              # Minimax + Alpha-Beta Pruning algorithm
│   └── model.ipynb        # Notebook for training neural network (AlphaZero style)
├── model/                 # Folder containing trained neural network models
└── README.md              # Project description file
</pre>

---

## 🧠 Implemented AI Algorithms

### 🔍 Minimax with Alpha-Beta Pruning

- Implements basic Minimax algorithm with **Alpha-Beta pruning** to reduce the number of nodes to traverse.
- Optimized for performance to play smoothly at a reasonable depth in real-time.
- Move evaluation is based on a **custom heuristic function**.

### 🌳 Monte Carlo Tree Search (MCTS)

- Standard MCTS implementation combined with node selection strategies such as **Upper Confidence Bound (UCB)**.
- Optimized for the number of rollouts and thinking time, suitable for real-time matches.

### 🧠 Neural Network - AlphaZero Style

- Uses **Reinforcement Learning (RL)** to train a neural network in AlphaZero style:
  - Employs **MCTS**, **Minimax**, and **self-play** to generate training data.
  - Network learns to **predict win probabilities** and **best moves** from a given state.
- Training is performed in `model.ipynb`, and the trained models are saved in the `model/` folder.

---
