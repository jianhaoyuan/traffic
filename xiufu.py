import os
import glob

# 👉 改成你自己的labels路径（复制粘贴你的路径）
LABEL_ROOT = r"dataset/labels"

def clean_invalid_labels():
    # 找到所有txt文件
    txt_files = glob.glob(os.path.join(LABEL_ROOT, "**/*.txt"), recursive=True)
    print(f"找到 {len(txt_files)} 个标注文件，开始清理错误ID...")

    for txt_path in txt_files:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            valid_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # 只保留：标准5列 + ID是0或1
                if len(parts) == 5 and parts[0] in ["0", "1"]:
                    valid_lines.append(line + "\n")

            # 覆盖保存
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(valid_lines)

        except Exception as e:
            continue

    print("✅ 所有错误标注ID已清理完成！")

if __name__ == "__main__":
    clean_invalid_labels()