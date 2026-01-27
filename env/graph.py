import json
import networkx as nx
import os

class TaxonomyGraph:
    def __init__(self, taxonomy_file_path):
        self.graph = nx.DiGraph()
        self.movie_details = {} 
        self.root = "ROOT"
        self.build_tree(taxonomy_file_path)

    def build_tree(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, dict): data = [data]

        self.graph.add_node(self.root, type='root', text="Movie Taxonomy Root")

        count = 0
        for item in data:
            movie_id = item.get('id')
            if movie_id is None: continue

            # --- CẬP NHẬT: Lưu thêm full_path từ JSON ---
            self.movie_details[movie_id] = {
                'title': item.get('title', 'Unknown'),
                'description': item.get('description', ''),
                'full_path': item.get('full_path', 'Unknown Path'), # Lấy path gốc
                'reasoning': item.get('taxonomy_data', {}).get('reasoning', '')
            }
            
            # Xây dựng cây (Logic giữ nguyên)
            tax_data = item.get('taxonomy_data', {})
            level_keys = [k for k in tax_data.keys() if k.startswith('level_')]
            sorted_levels = sorted(level_keys, key=lambda x: int(x.split('_')[1]))
            
            current_node = self.root
            for level_key in sorted_levels:
                category_name = tax_data[level_key]
                if not category_name: continue
                
                if not self.graph.has_node(category_name):
                    self.graph.add_node(category_name, type='category', text=category_name)
                
                if not self.graph.has_edge(current_node, category_name):
                    self.graph.add_edge(current_node, category_name)
                
                current_node = category_name
            
            if not self.graph.has_node(movie_id):
                self.graph.add_node(movie_id, type='movie', text=f"Movie: {item.get('title')}")
            
            if not self.graph.has_edge(current_node, movie_id):
                self.graph.add_edge(current_node, movie_id)
            
            count += 1
        print(f"Graph loaded: {count} movies, {self.graph.number_of_nodes()} nodes.")

    def get_children(self, node):
        if node not in self.graph: return []
        return sorted(list(self.graph.successors(node)), key=lambda x: str(x))

    def is_ancestor_of(self, node, target_id):
        if node == target_id: return True
        try:
            return nx.has_path(self.graph, node, target_id)
        except:
            return False

    def get_node_text(self, node):
        if node in self.graph.nodes:
            return self.graph.nodes[node].get('text', str(node))
        return str(node)

    # --- CẬP NHẬT: Hàm lấy text chi tiết cho prompt ---
    def get_movie_prompt_text(self, movie_id):
        info = self.movie_details.get(movie_id, {})
        # Format chuẩn: Title + Description + Taxonomy Path
        return (
            f"   - Title: {info.get('title', 'Unknown')}\n"
            f"     Description: {info.get('description', '')}\n"
            f"     Taxonomy Path: {info.get('full_path', 'N/A')}"
        )

    # ==========================================
    # ✅ HÀM MỚI: TÌM SỐ CON LỚN NHẤT (MAX BRANCHING FACTOR)
    # ==========================================
    def get_max_branching_factor(self):
        max_children = 0
        node_with_max_children = None

        for node in self.graph.nodes():
            # Lấy danh sách con của node hiện tại
            children = list(self.graph.successors(node))
            num_children = len(children)
            
            if num_children > max_children:
                max_children = num_children
                node_with_max_children = node
        
        return max_children, node_with_max_children

if __name__ == "__main__":
    # Đường dẫn file json taxonomy của bạn
    dummy_filename = "../taxonomy/taxonomy_with_ids.json" 

    if not os.path.exists(dummy_filename):
        print(f"⚠️ Không tìm thấy file: {dummy_filename}. Hãy sửa lại đường dẫn.")
    else:
        # Khởi tạo Graph
        tax_graph = TaxonomyGraph(dummy_filename)

        print("\n--- KIỂM TRA MAX BRANCHING FACTOR ---")
        max_branch, max_node = tax_graph.get_max_branching_factor()
        
        print(f"🔥 SỐ LƯỢNG CON LỚN NHẤT (Max Branching Factor): {max_branch}")
        print(f"📍 Node có nhiều con nhất là: '{max_node}'")
        
        print("\n⚠️ LƯU Ý QUAN TRỌNG CHO RL AGENT:")
        print(f"   Hãy đảm bảo tham số `output_dim` của mạng Neural Network >= {max_branch}")
        
        # --- TEST KHÁC ---
        # Kiểm tra logic hiển thị (đã sửa tên hàm cho đúng class bên trên)
        # Giả sử trong file json có movie id là 4 (nếu không có sẽ lỗi KeyError hoặc in ra None)
        try:
            detail_text = tax_graph.get_movie_prompt_text(4) 
            print(f"\nExample Movie Detail (ID=4):\n{detail_text}")
        except Exception as e:
            print("\n(Không test được ID=4 vì không có trong data mẫu)")