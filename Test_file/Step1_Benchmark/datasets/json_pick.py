import json
import random

# 输入文件路径
input_file = "arena_hard_v2.jsonl"
# 输出文件路径
output_file = "arena_hard_v2_picked.jsonl"
# 要随机选取的行数
sample_size = 100

# 读取所有行
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 如果文件行数少于 sample_size，则全部选取
sample_size = min(sample_size, len(lines))

# 随机选取指定数量的行
sampled_lines = random.sample(lines, sample_size)

# 写入新文件
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(sampled_lines)

print(f"已从 {input_file} 随机选取 {sample_size} 行写入 {output_file}")
