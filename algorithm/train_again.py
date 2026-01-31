import torch
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter

# --- 1. IMPORTS TỪ CODE CŨ ---
try:
    from algorithm.train_agent import (
        DoubleDQNAgent, 
        ReplayBuffer, 
        TaxonomyRLEnv, 
        evaluate_agent,
        TaxonomyGraph, 
        EmbeddingExtractor
    )
except ImportError:
    print("❌ Lỗi: Không tìm thấy file 'train_agent.py'.")
    exit()

# ==========================================
# 2. A100 OPTIMIZATION
# ==========================================
torch.set_float32_matmul_precision('high') 
torch.backends.cudnn.benchmark = True

# ==========================================
# 3. HÀM UPDATE MỚI (CHỈ DÙNG ONLINE DATA)
# ==========================================
def update_online_only(agent, online_buffer, batch_size):
    """
    Hàm update thay thế cho update_mixed vì không có Expert Data (do bỏ qua GĐ1).
    Chỉ sample từ Online Buffer.
    """
    if len(online_buffer) < batch_size:
        return None

    # Chỉ lấy mẫu từ Online Buffer
    state, action, reward, next_state, done = online_buffer.sample(batch_size)

    # Chuyển sang Tensor (A100 xử lý lô lớn ở đây)
    state = torch.FloatTensor(np.array(state)).to(agent.device)
    action = torch.LongTensor(np.array(action)).unsqueeze(1).to(agent.device)
    reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(agent.device)
    next_state = torch.FloatTensor(np.array(next_state)).to(agent.device)
    done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(agent.device)

    # Tính Q-Values hiện tại
    q_values = agent.policy_net(state)
    q_value = q_values.gather(1, action)

    # Tính Q-Values mục tiêu (Double DQN)
    with torch.no_grad():
        next_actions = agent.policy_net(next_state).argmax(1, keepdim=True)
        next_q_values = agent.target_net(next_state)
        next_q_value = next_q_values.gather(1, next_actions)
        expected_q_value = reward + agent.gamma * next_q_value * (1 - done)

    # Tính Loss & Backprop
    loss = agent.loss_fn(q_value, expected_q_value)
    
    agent.optimizer.zero_grad()
    loss.backward()
    # Clip grad để tránh vỡ gradient khi batch quá lớn
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
    agent.optimizer.step()
    
    return loss.item()

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    # --- CONFIG ---
    TAXONOMY_FILE = "taxonomy/taxonomy_with_ids.json"
    SESSIONS_FILE = "data/movie_sessions_ids_train.jsonl"
    EVAL_SESSIONS_FILE = "data/movie_sessions_ids_test.jsonl"
    MODEL_NAME = "Qwen/Qwen2-7B"
    CHECKPOINT_PATH = "ddqn_ep26000_acc0.25.pth"

    # --- HYPERPARAMETERS (A100) ---
    BATCH_SIZE = 4096      # Batch cực lớn cho A100
    MAX_EPISODES = 500_000 
    TARGET_UPDATE_FREQ = 1000 
    LR = 1e-4
    GAMMA = 0.99
    
    # Epsilon giảm chậm hơn (theo yêu cầu)
    EPSILON_START = 1.0   # Bắt đầu lại từ 1.0 để khám phá môi trường với kiến thức mới
    EPSILON_END = 0.05
    EPSILON_DECAY = 100_000 # Giảm rất chậm (trong 100k episode đầu)
    
    # Buffer
    ONLINE_BUFFER_SIZE = 1_000_000 # Tăng buffer lên 1 triệu (RAM hệ thống server thường lớn)

    # Eval
    EVAL_FREQ = 1000       
    EVAL_EPISODES = 1000    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(log_dir="./runs/DDQN_A100_Phase3_Only")

    print(f"--- Device: {DEVICE} (A100 Optimization Enabled) ---")

    # Init Components
    graph = TaxonomyGraph(TAXONOMY_FILE)
    embedder = EmbeddingExtractor(MODEL_NAME, DEVICE)
    
    train_env = TaxonomyRLEnv(graph, embedder, SESSIONS_FILE, depth_win=1)
    eval_env = TaxonomyRLEnv(graph, embedder, EVAL_SESSIONS_FILE, depth_win=1)
    
    agent = DoubleDQNAgent(train_env.observation_space.shape[0], train_env.action_space.n, lr=LR, gamma=GAMMA, device=DEVICE)
    
    # --- LOAD CHECKPOINT ---
    print(f"📥 Loading checkpoint: {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(checkpoint)
        print("✅ Checkpoint loaded! Weights restored.")
    except Exception as e:
        print(f"⚠️ Load failed: {e}. Starting fresh.")

    # Chỉ cần Online Buffer
    online_buffer = ReplayBuffer(ONLINE_BUFFER_SIZE)

    print(f"\n🚀 STARTING PHASE 3 DIRECTLY (Online Training)...")
    print(f"   Batch Size: {BATCH_SIZE} | Epsilon Decay: {EPSILON_DECAY} steps")

    epsilon = EPSILON_START
    total_steps = 0
    start_time = time.time()

    for episode in range(MAX_EPISODES):
        state, _ = train_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 1. Select Action
            action = agent.select_action(state, epsilon)
            
            # 2. Step
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            
            # 3. Store
            online_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # 4. Train (Update Online Only)
            # Chỉ train khi đủ 1 Batch Size A100 (4096 mẫu)
            if len(online_buffer) > BATCH_SIZE:
                loss = update_online_only(agent, online_buffer, BATCH_SIZE)
                if loss is not None: writer.add_scalar("Train/Loss", loss, total_steps)

            # 5. Update Target Net
            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
            
            # 6. Decay Epsilon
            # Công thức Exponential Decay: E = End + (Start - End) * exp(-steps / decay)
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * total_steps / EPSILON_DECAY)

        # Log
        writer.add_scalar("Train/Reward_Episode", episode_reward, episode)
        writer.add_scalar("Train/Epsilon", epsilon, episode)

        # --- EVALUATION ---
        if (episode + 1) % EVAL_FREQ == 0:
            avg_rew, acc = evaluate_agent(agent, eval_env, EVAL_EPISODES, DEVICE)
            writer.add_scalar("Eval/Accuracy", acc, episode)
            writer.add_scalar("Eval/AvgReward", avg_rew, episode)
            
            elapsed = time.time() - start_time
            print(f"Ep {episode+1} | Acc: {acc*100:.2f}% | Epsilon: {epsilon:.4f} | Time: {elapsed:.0f}s")
            
            # Save
            torch.save(agent.policy_net.state_dict(), f"phase3_retrain_ep{episode+1}_acc{acc:.2f}.pth")
            start_time = time.time()

    writer.close()
