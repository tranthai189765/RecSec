## Lệnh cài thư viện (Python 3.10)

```bash
pip install torch numpy pandas networkx transformers gymnasium tensorboard
```


## Lệnh chạy train model

```bash
python -m algorithm.train_agent
```

## 🧠 Reinforcement Learning Environment: `TaxonomyRLEnv` (Qwen2-7B)

### 🔍 Overview
`TaxonomyRLEnv` là một môi trường Reinforcement Learning (Gymnasium) cho bài toán **taxonomy navigation** trong recommendation.
Agent có nhiệm vụ điều hướng trong **cây taxonomy phim**, dựa trên **lịch sử xem phim của người dùng**, nhằm xác định đúng **category hoặc movie target**.

Agent bắt đầu từ **root node** và chọn node con từng bước.  
Trạng thái được biểu diễn bằng **LLM-based semantic embedding**.

---

### 📥 Inputs

| Thành phần | Mô tả |
|----------|------|
| `taxonomy_tree` | Cây taxonomy phim (graph / tree) |
| `sessions_file` | File `.jsonl` chứa user watch sessions |
| `embedding_model` | Mô hình sinh embedding từ prompt |
| `movie_details` | Metadata phim (title, description) |

---

### ⚙️ Environment Parameters

| Parameter | Type | Description |
|---------|------|-------------|
| `taxonomy_tree` | `TaxonomyGraph` | Cây taxonomy phim |
| `embedding_model` | `EmbeddingExtractor` | LLM-based embedding model |
| `sessions_file` | `str` | Đường dẫn file session |
| `depth_win` | `int` / `None` | Độ sâu tối đa để được tính là thắng |

#### `depth_win` Logic
- `depth_win = None`: Phải đến **đúng movie target**
- `depth_win = k`: Chỉ cần đi đúng đường đến **depth ≥ k**
- Nếu target nông hơn `depth_win`: Phải đến **đúng target**

---

### 🎮 Action Space

| Property | Value |
|--------|------|
| Type | `Discrete` |
| Size | `32` |
| Meaning | Chọn node con tiếp theo trong taxonomy |

📌 Nếu số node con < 32 → action sẽ được **clip** về action hợp lệ.

---

### 👁️ Observation Space

| Property | Value |
|--------|------|
| Type | `Box` |
| Shape | `(4096,)` |
| Dtype | `float32` |
| Content | Semantic embedding của trạng thái |

Observation được sinh từ:
- User watch history (reveal theo depth)
- Trajectory hiện tại trong taxonomy
- Danh sách node con khả dụng
- System instruction cho agent

---

### 🏁 Episode Flow

| Step | Description |
|----|-------------|
| `reset()` | Chọn ngẫu nhiên session, reset về root |
| `step(action)` | Di chuyển đến node con |
| `terminated=True` | Khi thắng, sai đường hoặc node lá |

---

### 🧮 Reward Function

| Situation | Reward | Terminated |
|----------|--------|------------|
| Correct category step | `+0.1` | ❌ |
| Reach exact movie target | `+1.0` | ✅ |
| Reach `depth_win` correctly | `+1.0` | ✅ |
| Wrong path | `-1.0` | ✅ |
| No children available | `0.0` | ✅ |

---

### ℹ️ Info Field

| Key | Meaning |
|---|--------|
| `success_found_movie` | Đến đúng movie |
| `success_depth_limit_reached` | Thắng do đạt depth |
| `correct_step_category` | Đi đúng category |
| `wrong_path` | Đi sai nhánh |
| `no_children_available` | Node lá |
| `action_clipped` | Action vượt giới hạn |

---
