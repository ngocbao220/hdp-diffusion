import os
import pathlib

def scan_configs(start_path='configs'):
    if not os.path.exists(start_path):
        print(f"❌ Error: Folder '{start_path}' not found in current directory.")
        print(f"Current directory is: {os.getcwd()}")
        return

    print(f"📂 Scanning structure for '{start_path}'...\n")
    
    overrides_suggestion = {}
    
    # Duyệt cây thư mục
    for root, dirs, files in os.walk(start_path):
        # Tính độ sâu để in thụt đầu dòng
        level = root.replace(start_path, '').count(os.sep)
        indent = '│   ' * level
        folder_name = os.path.basename(root)
        
        # Bỏ qua folder chính khi in tên
        if root != start_path:
            print(f"{indent}├── 📁 {folder_name}/")
            sub_indent = '│   ' * (level + 1)
        else:
            print(f"📁 {folder_name}/ (ROOT)")
            sub_indent = '│   '
            
        # Lọc file yaml
        yaml_files = [f for f in files if f.endswith('.yaml') or f.endswith('.yml')]
        
        for f in sorted(yaml_files):
            print(f"{sub_indent}├── 📄 {f}")
            
            # Logic gợi ý override cho Hydra
            # Nếu file nằm trong configs/algo/name.yaml -> algo=name
            # Nếu file nằm trong configs/data/name.yaml -> data=name
            if root != start_path:
                group_name = folder_name # VD: algo, data, model
                option_name = os.path.splitext(f)[0] # VD: bd3lm, gsm8k
                
                if group_name not in overrides_suggestion:
                    overrides_suggestion[group_name] = []
                overrides_suggestion[group_name].append(option_name)

    print("\n" + "="*50)
    print("💡 GỢI Ý CÁC OPTIONS CÓ THỂ DÙNG (OVERRIDES):")
    print("="*50)
    
    for group, options in overrides_suggestion.items():
        print(f"\n🔹 Group: {group} (dùng {group}=...)")
        print(f"   Options: {', '.join(options)}")

if __name__ == "__main__":
    scan_configs()