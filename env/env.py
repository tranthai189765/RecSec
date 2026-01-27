import json
import networkx as nx
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.graph import TaxonomyGraph
from env.model import EmbeddingExtractor
import os

class TaxonomyRLEnv(gym.Env):
    def __init__(self, taxonomy_tree, embedding_model, sessions_file, depth_win=None):
        """
        depth_win (int): Độ sâu tối đa để tính là chiến thắng. 
                         Nếu target sâu hơn depth_win, chỉ cần đến depth_win là thắng.
                         Nếu target nông hơn depth_win, phải đến đúng target.
        """
        super(TaxonomyRLEnv, self).__init__()
        
        self.tree = taxonomy_tree
        self.embedder = embedding_model
        self.depth_win = depth_win  # <--- NEW PARAMETER
        
        self.sessions = []
        if not os.path.exists(sessions_file):
            raise FileNotFoundError(f"File not found: {sessions_file}")

        print(f"Loading sessions from {sessions_file}...")
        with open(sessions_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sess = json.loads(line)
                    if sess['label'] in self.tree.movie_details:
                        self.sessions.append(sess)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(self.sessions)} valid sessions.")
        if len(self.sessions) == 0:
            raise ValueError("No valid sessions found.")

        self.max_children = 32 
        self.action_space = spaces.Discrete(self.max_children)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4096,), dtype=np.float32)
        
        self.current_session = None
        self.target_label = None
        self.current_node = None
        self.path_history = []
        
    def _get_movie_full_path(self, movie_id):
        try:
            path = nx.shortest_path(self.tree.graph, source=self.tree.root, target=movie_id)
            return path
        except (nx.NetworkXNoPath, Exception):
            return [self.tree.root, "Unknown", movie_id]

    def _get_node_display_text(self, node_id):
        if node_id in self.tree.movie_details:
            info = self.tree.movie_details[node_id]
            title = info.get('title', 'Unknown Title')
            desc = info.get('description', 'No description')
            return f'Movie: "{title}" - {desc}'
        else:
            return f'Category: "{node_id}"'

    def _get_observation(self):
        system_prompt = (
            "SYSTEM INSTRUCTION: You are a movie taxonomy navigation agent. "
            "Your task is to identify the correct category for a target user session "
            "by traversing the taxonomy tree based on their watch history."
        )

        current_depth = len(self.path_history)
        history_parts = []
        
        for idx, movie_id in enumerate(self.current_session['session']):
            movie_path = self._get_movie_full_path(movie_id)
            reveal_levels = current_depth
            is_full_reveal = (reveal_levels >= len(movie_path) - 1) or (current_depth >= 4)

            readable_path = []
            for node in movie_path:
                if node in self.tree.movie_details:
                     readable_path.append(self.tree.movie_details[node].get('title', str(node)))
                else:
                     readable_path.append(str(node))

            if is_full_reveal:
                movie_text = self.tree.get_movie_prompt_text(movie_id).strip()
                path_str = " -> ".join(readable_path)
                history_parts.append(f"{idx+1}. {movie_text} || [Full Path]: {path_str}")
            else:
                visible_nodes = readable_path[1 : reveal_levels + 1]
                if visible_nodes:
                    cat_text = " > ".join(visible_nodes)
                    history_parts.append(f"{idx+1}. [Categories Level 1-{reveal_levels}]: {cat_text} > ...")
                else:
                    history_parts.append(f"{idx+1}. [Category]: Unknown")

        history_text = "\n".join(history_parts)
        path_str_list = [str(node) for node in self.path_history]
        path_text = " -> ".join(path_str_list)
        
        children = self.tree.get_children(self.current_node)
        children_descriptions = []
        for i, child in enumerate(children):
            display_text = self._get_node_display_text(child)
            children_descriptions.append(f"Action {i}: {display_text}")
        
        if not children_descriptions:
            children_text = "No available nodes (Leaf node reached)"
        else:
            children_text = "\n".join(children_descriptions)
        
        full_text = (
            f"{system_prompt}\n\n"
            f"=== USER WATCH HISTORY (Depth-based Detail: Level {current_depth}) ===\n"
            f"{history_text}\n\n"
            f"=== CURRENT TRAJECTORY ===\n"
            f"{path_text}\n\n"
            f"=== AVAILABLE NEXT NODES (Choose Action Index) ===\n"
            f"{children_text}"
        )
        
        # DEBUG
        # print(f"\n[DEBUG PROMPT]:\n{full_text}...\n") 

        embedding = self.embedder.get_embedding(full_text)
        return embedding

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_session = self.sessions[np.random.randint(len(self.sessions))]
        self.target_label = self.current_session['label']
        
        self.current_node = self.tree.root
        self.path_history = [self.tree.root]
        
        return self._get_observation(), {}

    def step(self, action):
        children = self.tree.get_children(self.current_node)
        
        terminated = False
        truncated = False
        reward = 0
        info = {}
        
        if len(children) == 0:
            terminated = True
            info = {"result": "no_children_available"}
        else:
            # --- Action Clipping ---
            if action >= len(children):
                action = len(children) - 1
                info['note'] = "action_clipped"

            selected_node = children[action]
            
            # --- Check Path Correctness ---
            if self.tree.is_ancestor_of(selected_node, self.target_label):
                self.current_node = selected_node
                self.path_history.append(selected_node)
                
                # Tính độ sâu hiện tại (Trừ 1 vì Root không tính là bước đi)
                # path_history = [Root, L1, L2] -> len=3 -> current_depth_level = 2
                current_depth_level = len(self.path_history) - 1
                
                # --- WIN CONDITION LOGIC ---
                is_exact_target = (selected_node == self.target_label)
                is_depth_win = (self.depth_win is not None and current_depth_level >= self.depth_win)

                if is_exact_target:
                    # Case 1: Tìm thấy chính xác phim (dù depth < depth_win hay >= depth_win đều thắng)
                    terminated = True
                    reward = 1.0
                    info["result"] = "success_found_movie"
                elif is_depth_win:
                    # Case 2: Chưa đến phim, nhưng đã chạm mốc depth_win và đang đi đúng đường
                    terminated = True
                    reward = 1.0
                    info["result"] = "success_depth_limit_reached"
                else:
                    # Case 3: Đi đúng đường nhưng chưa đến đích và chưa đủ sâu
                    reward = 0.1
                    info["result"] = "correct_step_category"
            else:
                # --- Wrong Path ---
                reward = -1
                terminated = True
                info["result"] = "wrong_path"
                
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    TAXONOMY_FILE = "taxonomy/taxonomy_with_ids.json"
    SESSIONS_FILE = "data/movie_sessions_ids.jsonl"

    print("--- 1. Init Graph ---")
    try:
        graph = TaxonomyGraph(TAXONOMY_FILE)
    except Exception:
        print("Error loading graph")
        exit()

    print("--- 2. Init Model ---")
    MODEL_NAME = "Qwen/Qwen2-7B"
    try:
        embedder = EmbeddingExtractor(MODEL_NAME, "cuda")
    except Exception:
        exit()

    print("--- 3. Run Test with DEPTH WIN = 2 ---")
    # THỬ NGHIỆM VỚI DEPTH_WIN = 2 (Chỉ cần đoán đúng đến Level 2 là thắng)
    env = TaxonomyRLEnv(graph, embedder, SESSIONS_FILE, depth_win=1)
    
    obs, _ = env.reset()
    
    # Lấy thông tin target để cheat (test)
    target_id = env.target_label
    target_path = nx.shortest_path(graph.graph, graph.root, target_id)
    # Ví dụ path: [Root, Action, Superhero, IronMan]
    
    print(f"🎯 Target Full Path: {target_path}")
    print(f"⚠️ Win Condition: Reach ID {target_id} OR Reach Depth 2 (Node: {target_path[2] if len(target_path)>2 else 'N/A'})")

    terminated = False
    step = 0
    while not terminated:
        step += 1
        children = graph.get_children(env.current_node)
        
        # Oracle Action (Luôn chọn đúng)
        correct_action = -1
        for i, child in enumerate(children):
            if graph.is_ancestor_of(child, target_id):
                correct_action = i
                break
        
        print(f"\n--- Step {step} ---")
        if correct_action != -1:
            print(f"🤖 Oracle selecting action {correct_action} (Node: {children[correct_action]})")
            obs, reward, terminated, truncated, info = env.step(correct_action)
            print(f"   Result: {info['result']}")
            print(f"   Reward: {reward}")
            print(f"   Terminated: {terminated}")
        else:
            print("❌ Oracle lost path!")
            break
        
        if terminated:
            print("\n🏁 EPISODE FINISHED 🏁")