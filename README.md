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


## 🤖 Learning Algorithm: Double DQN with Dual Replay Buffers

### 🔍 Algorithm Overview
Thuật toán sử dụng **Double DQN kết hợp Dueling Architecture**, được thiết kế cho bài toán **taxonomy navigation** với không gian hành động rời rạc và trạng thái embedding chiều cao.

Đặc điểm chính:
- **Double DQN**: giảm overestimation bias
- **Dueling Network**: tách Value và Advantage
- **Dual Replay Buffers**:
  - Expert Buffer (oracle-guided)
  - Online Buffer (agent interaction)
- **Curriculum Learning**: giảm dần sự phụ thuộc vào expert

---

### 🧠 Network Architecture

| Component | Description |
|--------|-------------|
| Backbone | Dueling Deep Q-Network |
| Input | State embedding `(4096,)` |
| Output | Q-values cho từng action |
| Heads | Value stream + Advantage stream |
| Target Network | Cập nhật định kỳ |

---

### 🧮 Reinforcement Learning Setup

| Element | Definition |
|------|-----------|
| State (S) | LLM-based semantic embedding |
| Action (A) | Chọn node con trong taxonomy |
| Reward (R) | Shaped reward theo path |
| Transition | `(s, a, r, s', done)` |
| Objective | Maximize expected cumulative reward |

---

### 🎮 Action Selection

| Strategy | Description |
|-------|-------------|
| Exploration | Epsilon-Greedy |
| Exploitation | Greedy action (`argmax Q`) |
| Oracle | Expert action dựa trên taxonomy path |

---

### 📦 Replay Buffers

| Buffer | Size | Content |
|------|------|--------|
| Expert Buffer | `50,000` | Oracle-guided transitions |
| Online Buffer | `100,000` | Agent self-exploration |

---

### ⚙️ Training Hyperparameters

| Parameter | Value |
|--------|------|
| Batch Size | `256` |
| Learning Rate | `1e-4` |
| Discount Factor (γ) | `0.99` |
| Target Update Frequency | `500` steps |
| Max Episodes | `500,000` |

---

### 🎯 Exploration Parameters

| Parameter | Value |
|---------|------|
| Epsilon Start | `1.0` |
| Epsilon End | `0.05` |
| Epsilon Decay | `10,000` steps |

---

### 🧪 Expert Mixing Strategy

| Parameter | Description |
|---------|-------------|
| `expert_ratio_start` | `0.5` |
| `expert_ratio_end` | `0.1` |
| `expert_decay_eps` | `5,000` episodes |

➡️ Tỷ lệ sample từ expert buffer **giảm dần theo episode**.

---

### 🏗️ Training Phases

#### Phase 1 — Expert Prefill
| Aspect | Value |
|-----|------|
| Steps | `100,000` |
| Policy | Oracle (perfect navigation) |
| Buffer | Expert Buffer |

---

#### Phase 2 — Pre-training
| Aspect | Value |
|------|------|
| Updates | `200,000` |
| Data | Expert Buffer only |
| Objective | Warm-start Q-network |

---

#### Phase 3 — Online Training
| Aspect | Description |
|-----|------------|
| Policy | Epsilon-Greedy |
| Buffer | Mixed (Expert + Online) |
| Target Net | Periodic sync |
| Logging | TensorBoard |

---

### 📊 Evaluation Protocol

| Parameter | Value |
|--------|------|
| Eval Frequency | Every `1000` episodes |
| Eval Episodes | `100` |
| Policy | Greedy (ε = 0) |
| Metrics | Avg Reward, Accuracy |

**Success Conditions**:
- `success_found_movie`
- `success_depth_limit_reached`

---

### 🧮 Loss Function

| Component | Description |
|--------|-------------|
| Loss | Mean Squared Error (MSE) |
| Target | Double DQN target |
| Gradient Clipping | `max_norm = 1.0` |

---

### 📈 Logged Metrics (TensorBoard)

| Metric | Description |
|-----|-------------|
| Train/Loss | Q-learning loss |
| Train/Reward | Episode reward |
| Train/Epsilon | Exploration rate |
| Eval/Accuracy | Success rate |
| Eval/AvgReward | Avg reward on test |

---
