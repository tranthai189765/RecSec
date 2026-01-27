import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# --- 1. IMPORTS ---
from algorithm.dueling_model import DuelingDQN

try:
    from env.env import TaxonomyRLEnv 
    from env.model import EmbeddingExtractor
    from env.graph import TaxonomyGraph
except ImportError as e:
    print(f"⚠️ Lỗi Import: {e}")
    exit()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_oracle_action(env):
    """Chọn action đúng 100% dựa trên graph structure"""
    children = env.tree.get_children(env.current_node)
    for idx, child in enumerate(children):
        if env.tree.is_ancestor_of(child, env.target_label):
            return idx
    return random.randrange(len(children)) if len(children) > 0 else 0

def evaluate_agent(agent, env, num_episodes, device):
    """
    Hàm đánh giá Agent trên tập Test (Evaluation Environment)
    - Không dùng Epsilon-Greedy (chọn best action)
    - Không update weights
    """
    agent.policy_net.eval() # Chuyển sang chế độ eval (tắt dropout nếu có)
    total_reward = 0
    success_count = 0
    
    # Định nghĩa các trạng thái thành công
    SUCCESS_RESULTS = ["success_found_movie", "success_depth_limit_reached"]

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select Greedy Action (Epsilon = 0)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax().item()
            
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if done and info.get("result") in SUCCESS_RESULTS:
                success_count += 1
                
        total_reward += episode_reward

    agent.policy_net.train() # Chuyển lại về chế độ train
    
    avg_reward = total_reward / num_episodes
    accuracy = success_count / num_episodes
    return avg_reward, accuracy

# ==========================================
# 3. REPLAY BUFFER
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 4. DOUBLE DQN AGENT
# ==========================================
class DoubleDQNAgent:
    def __init__(self, input_dim, output_dim, lr=1e-4, gamma=0.99, device="cuda"):
        self.device = device
        self.gamma = gamma
        self.output_dim = output_dim

        self.policy_net = DuelingDQN(input_dim, output_dim).to(device)
        self.target_net = DuelingDQN(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.output_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update_mixed(self, expert_buffer, online_buffer, batch_size, expert_ratio):
        n_expert = int(batch_size * expert_ratio)
        n_online = batch_size - n_expert

        if len(online_buffer) < n_online:
            return None
        if len(expert_buffer) < n_expert:
            n_expert = len(expert_buffer)

        s_e, a_e, r_e, ns_e, d_e = expert_buffer.sample(n_expert)
        s_o, a_o, r_o, ns_o, d_o = online_buffer.sample(n_online)

        state = torch.FloatTensor(np.concatenate((s_e, s_o))).to(self.device)
        action = torch.LongTensor(np.concatenate((a_e, a_o))).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(np.concatenate((r_e, r_o))).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.concatenate((ns_e, ns_o))).to(self.device)
        done = torch.FloatTensor(np.concatenate((d_e, d_o))).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state)
        q_value = q_values.gather(1, action)

        with torch.no_grad():
            next_actions = self.policy_net(next_state).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.gather(1, next_actions)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = self.loss_fn(q_value, expected_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    TAXONOMY_FILE = "taxonomy/taxonomy_with_ids.json"
    SESSIONS_FILE = "data/movie_sessions_ids_train.jsonl"
    EVAL_SESSIONS_FILE = "data/movie_sessions_ids_test.jsonl" # File dành riêng cho Eval
    MODEL_NAME = "Qwen/Qwen2-7B"

    BATCH_SIZE = 256
    
    # --- THAY ĐỔI 1: Tăng thời gian train Part 3 ---
    MAX_EPISODES = 500_000 
    
    TARGET_UPDATE_FREQ = 500 
    LR = 1e-4
    GAMMA = 0.99
    
    # --- THAY ĐỔI 2: Epsilon giảm nhanh hơn ---
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 10_000 
    
    # Config Buffer
    EXPERT_BUFFER_SIZE = 50_000
    ONLINE_BUFFER_SIZE = 100_000
    PREFILL_STEPS = 100_000      
    PRETRAIN_UPDATES = 200_000    
    EXPERT_RATIO_START = 0.5 
    EXPERT_RATIO_END = 0.1     
    EXPERT_DECAY_EPS = 5_000

    # --- CONFIG EVAL ---
    EVAL_FREQ = 1000       # Cứ 1000 Episode train thì Eval 1 lần
    EVAL_EPISODES = 100    # Mỗi lần Eval chạy 100 episode để lấy trung bình

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(log_dir="./runs/DDQN_DualBuffer_Eval")

    # 1. Init Graph & Embedder (Dùng chung)
    print("--- Initializing Graph & Embedder ---")
    graph = TaxonomyGraph(TAXONOMY_FILE)
    embedder = EmbeddingExtractor(MODEL_NAME, DEVICE)

    # 2. Init Env (Train & Eval riêng biệt)
    print("--- Initializing Environments ---")
    # depth_win=1 để train cho dễ, hoặc chỉnh lên nếu muốn khó hơn
    train_env = TaxonomyRLEnv(graph, embedder, SESSIONS_FILE, depth_win=1)
    
    # --- THAY ĐỔI 3: Tạo Eval Env ---
    eval_env = TaxonomyRLEnv(graph, embedder, EVAL_SESSIONS_FILE, depth_win=1)
    
    input_dim = train_env.observation_space.shape[0]
    output_dim = train_env.action_space.n 
    agent = DoubleDQNAgent(input_dim, output_dim, lr=LR, gamma=GAMMA, device=DEVICE)
    
    expert_buffer = ReplayBuffer(EXPERT_BUFFER_SIZE)
    online_buffer = ReplayBuffer(ONLINE_BUFFER_SIZE)

    # ==========================================
    # GIAI ĐOẠN 1: PRE-FILL EXPERT BUFFER
    # ==========================================
    print(f"\n🧠 GIAI ĐOẠN 1: Expert Prefill ({PREFILL_STEPS} steps)...")
    state, _ = train_env.reset()
    for i in range(PREFILL_STEPS):
        action = get_oracle_action(train_env)
        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated
        expert_buffer.push(state, action, reward, next_state, done)
        state = next_state
        if done: state, _ = train_env.reset()
        if (i+1) % 100 == 0: print(f"   -> Prefilled {i+1} transitions.")

    # ==========================================
    # GIAI ĐOẠN 2: PRE-TRAIN
    # ==========================================
    print(f"\n💪 GIAI ĐOẠN 2: Pre-training ({PRETRAIN_UPDATES} updates)...")
    for i in range(PRETRAIN_UPDATES):
        loss = agent.update_mixed(expert_buffer, expert_buffer, BATCH_SIZE, expert_ratio=0.5)
        if (i+1) % 2000 == 0: print(f"   -> Pretrain loss: {loss:.6f}")

    # ==========================================
    # GIAI ĐOẠN 3: ONLINE TRAINING + EVAL
    # ==========================================
    print(f"\n🚀 GIAI ĐOẠN 3: Online Training (Max Episodes: {MAX_EPISODES})...")
    
    epsilon = EPSILON_START
    total_steps = 0

    for episode in range(MAX_EPISODES):
        state, _ = train_env.reset()
        episode_reward = 0
        done = False
        
        # Decay Expert Ratio
        ratio_decay_factor = max(0, (1 - episode / EXPERT_DECAY_EPS))
        current_expert_ratio = EXPERT_RATIO_END + (EXPERT_RATIO_START - EXPERT_RATIO_END) * ratio_decay_factor

        while not done:
            # Select Action
            action = agent.select_action(state, epsilon)
            
            # Step
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            # Store & Update
            online_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if len(online_buffer) > BATCH_SIZE:
                loss = agent.update_mixed(expert_buffer, online_buffer, BATCH_SIZE, current_expert_ratio)
                if loss is not None: writer.add_scalar("Train/Loss", loss, total_steps)

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
            
            # Epsilon Decay (Nhanh hơn)
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * total_steps / EPSILON_DECAY)

        # Log Training Metrics
        writer.add_scalar("Train/Reward_Episode", episode_reward, episode)
        writer.add_scalar("Train/Epsilon", epsilon, episode)
        
        # --- THAY ĐỔI 4: LOGIC EVALUATION ---
        if (episode + 1) % EVAL_FREQ == 0:
            print(f"🔄 Starting Evaluation at Episode {episode+1}...")
            avg_eval_reward, eval_accuracy = evaluate_agent(agent, eval_env, EVAL_EPISODES, DEVICE)
            
            # Log to Tensorboard
            writer.add_scalar("Eval/AvgReward", avg_eval_reward, episode)
            writer.add_scalar("Eval/Accuracy", eval_accuracy, episode)
            
            print(f"   📊 Eval Result: Accuracy={eval_accuracy*100:.2f}% | AvgReward={avg_eval_reward:.4f}")
            
            # Save Model Checkpoint
            torch.save(agent.policy_net.state_dict(), f"ddqn_ep{episode+1}_acc{eval_accuracy:.2f}.pth")

        # Logging Text (Training)
        if (episode+1) % 100 == 0:
            print(f"Ep {episode+1} | TrainRew: {episode_reward:.2f} | Eps: {epsilon:.3f} | Result: {info.get('result')}")

    writer.close()
    print("Training Complete!")