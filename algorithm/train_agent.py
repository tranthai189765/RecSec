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
