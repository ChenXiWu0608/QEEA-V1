import os
import pandas as pd

# 1️⃣ 配置文件夹路径
CSV_DIR = "reports2"  # 替换为你的 CSV 文件夹
OUTPUT_FILE = "timeout_tasks.csv"

# 2️⃣ 125 秒阈值（毫秒）
THRESHOLD_MS = 125 * 1000

# 3️⃣ 遍历文件夹下所有 CSV
summary_rows = []

for file_name in os.listdir(CSV_DIR):
    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(CSV_DIR, file_name)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"⚠️ 读取文件失败: {file_name}, {e}")
        continue

    # 检查必须存在的列
    if not {'task_id', 'workflow_name', 'question_index', 'start_time_ms', 'end_time_ms'}.issubset(df.columns):
        print(f"⚠️ 文件缺少必要列: {file_name}")
        continue

    # 遍历每一行，计算时间差
    for idx, row in df.iterrows():
        start_ms = row['start_time_ms']
        end_ms = row['end_time_ms']
        time_diff = end_ms - start_ms

        # 如果接近 125 秒，就记录
        if abs(time_diff - THRESHOLD_MS) < 3000:  # 误差 1 秒以内
            summary_rows.append({
                "file_name": file_name,
                "workflow_name": row['workflow_name'],
                "task_id": row['task_id'],
                "question_index": str(row['question_index']),
                "start_time_ms": start_ms,
                "end_time_ms": end_ms,
                "actual_duration_sec": ""  # 留空，手动填写
            })

# 4️⃣ 输出结果表格
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 已生成超时任务汇总: {OUTPUT_FILE}")
else:
    print("ℹ️ 未发现接近 125 秒的任务。")
