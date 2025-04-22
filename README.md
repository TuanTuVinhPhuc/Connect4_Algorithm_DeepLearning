# 🎮 CompetitionAI

---

## 🧠 Giới thiệu
Đây là một trò chơi **Connect4 cổ điển**, nơi bạn có thể thi đấu với nhiều loại AI khác nhau: từ **Minimax**, **Heuristic**, **MCTS**, cho tới **mạng Neural đơn giản**.

Dự án được chia thành **2 phần chính**:

- **Phần giao diện**: Quản lý bàn cờ, khởi tạo, và hiển thị game (sử dụng **Pygame** hoặc thư viện đồ họa khác).
- **Phần AI**: Bao gồm các thuật toán trí tuệ nhân tạo giúp máy tính đưa ra nước đi tối ưu.

---

## 📁 Cấu trúc thư mục
<pre>
connect4/
├── src/
│   ├── setting.py         # Thông số cố định của bàn cờ (kích thước, màu sắc, v.v.)
│   ├── board.py           # Logic khởi tạo và cập nhật bàn cờ
│   ├── ultis.py           # Các hàm hỗ trợ thao tác và cập nhật bàn cờ
│   ├── game.py            # Giao diện đấu giữa con người và AI
│   ├── ai_battle.py       # AI đấu với AI để so sánh sức mạnh các thuật toán
│   ├── MCTS.py            # Thuật toán Monte Carlo Tree Search
│   ├── ai.py              # Thuật toán Minimax + Alpha-Beta Pruning
│   ├── heuristic.py       # Hàm lượng giá heuristic đơn giản
│   └── model.ipynb        # Notebook huấn luyện mạng neural (AlphaZero style)
├── model/                 # Thư mục chứa các model mạng neural đã huấn luyện
└── README.md              # File mô tả dự án này
</pre>
---

## 🧠 Thuật toán AI đã triển khai

### 🔍 Minimax với Alpha-Beta Pruning

- Triển khai thuật toán Minimax cơ bản, kết hợp với **cắt tỉa Alpha-Beta** để giảm thiểu số node cần duyệt.
- Được tối ưu hóa về hiệu suất để chơi mượt ở độ sâu phù hợp trong thời gian thực.
- Có thể đánh giá nước đi dựa trên **hàm lượng giá heuristic** tuỳ chỉnh.

### 🌳 Monte Carlo Tree Search (MCTS)

- Cài đặt MCTS chuẩn, kết hợp với các chiến lược chọn node như **Upper Confidence Bound (UCB)**.
- Đã được tối ưu về số lần rollout và thời gian suy nghĩ, phù hợp để thi đấu real-time.

### 🧠 Mạng Neural - Phong cách AlphaZero

- Áp dụng **Reinforcement Learning (RL)** để huấn luyện mạng neural theo phong cách AlphaZero:
  - Dùng thuật toán **MCTS**, **Minimax** và **chính nó trong quá khứ** làm đối thủ sinh dữ liệu tự học.
  - Mạng học để **dự đoán xác suất thắng** và **hành động tốt nhất** từ trạng thái.
- Quá trình training được thực hiện trong `model.ipynb`, và mô hình được lưu tại thư mục `model/`.

---