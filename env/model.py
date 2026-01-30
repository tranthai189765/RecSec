import json
import networkx as nx
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- Class của bạn ---
class EmbeddingExtractor:
    def __init__(self, model_name, device):
        print(f"Loading LLM: {model_name}...")
        print(f"Device configuration: {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Nếu dùng Llama, cần set pad_token vì mặc định nó không có
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        # Lưu ý: Nếu GPU yếu, có thể thêm load_in_8bit=True (cần thư viện bitsandbytes)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_hidden_states=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            # load_in_8bit=True # Bỏ comment dòng này nếu VRAM < 14GB
        )
        self.device = device
        self.model.eval()

    def get_embedding(self, text):
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.tokenizer.model_max_length,
            padding=True # Nên thêm padding=True để an toàn
        ).to(self.device if self.device == "cuda" else "cpu") # model load auto device map handles placement, inputs need to match
        
        # Vì device_map="auto" sẽ tự phân bổ model lên GPU, 
        # nên input phải đưa vào device của layer đầu tiên của model
        if self.device == "cuda" and hasattr(self.model, "device"):
             inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Lấy hidden states
        hidden_states = outputs.hidden_states
        
        # Lấy layer ở giữa (Middle Layer)
        num_layers = len(hidden_states)
        middle_layer_idx = num_layers // 2
        
        # hidden_states là tuple, lấy phần tử tại index
        middle_layer_tensor = hidden_states[middle_layer_idx] # Shape: [batch_size, seq_len, hidden_size]
        
        # Mean Pooling để ra 1 vector đại diện
        # Lưu ý: cần xử lý attention mask nếu muốn chính xác tuyệt đối (bỏ qua padding token)
        # Nhưng ở đây test đơn giản nên mean trực tiếp cũng được.
        embedding = torch.mean(middle_layer_tensor, dim=1).squeeze().cpu().numpy()
        return embedding

# --- Phần chạy Test ---
if __name__ == "__main__":
    # 1. Cấu hình
    # Bạn có thể đổi thành "TinyLlama/TinyLlama-1.1B-Chat-v1.0" để test nhanh logic
    MODEL_NAME = "Qwen/Qwen2-7B"
    
    # Kiểm tra xem có GPU không
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Bắt đầu test trên device: {DEVICE} ---")

    try:
        # 2. Khởi tạo extractor
        extractor = EmbeddingExtractor(MODEL_NAME, DEVICE)
        
        # 3. Dữ liệu test
        test_texts = [
            "Hello world, this is a test for Llama embedding.",
            "Machine learning is fascinating."
        ]

        print("\n--- Đang trích xuất embedding ---")
        for text in test_texts:
            emb = extractor.get_embedding(text)
            
            print(f"\nInput Text: '{text}'")
            print(f"Embedding Shape: {emb.shape}")
            print(f"First 5 values: {emb[:5]}")
            
            # Kiểm tra xem có phải vector không (Llama 7B thường là size 4096)
            if emb.shape[0] == 4096:
                print(">> KẾT QUẢ: OK (Kích thước đúng với Llama 7B - 4096)")
            else:
                print(f">> KẾT QUẢ: OK (Kích thước hidden size: {emb.shape[0]})")

    except Exception as e:
        print(f"\n!!! LỖI XẢY RA !!!")
        print(e)
        print("\nGợi ý khắc phục:")
        print("- Nếu lỗi OutOfMemory (OOM): Hãy thử load model 8-bit hoặc dùng model nhỏ hơn.")
        print("- Nếu lỗi HuggingFace Auth: Hãy chạy `huggingface-cli login` trong terminal.")
