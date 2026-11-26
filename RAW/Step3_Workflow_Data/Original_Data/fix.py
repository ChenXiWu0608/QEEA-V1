import os
import pandas as pd

# -----------------------
# 配置
# -----------------------
CSV_DIR = "reports2"  # 原始任务报告 CSV 文件夹
TIMEOUT_SUMMARY = "timeout_tasks_summary2.csv"  # 你已填充实际耗时的表格
OUTPUT_DIR = "reports2_fixed"  # 修正后的 CSV 保存路径
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# 读取超时任务汇总
# -----------------------
timeout_df = pd.read_csv(TIMEOUT_SUMMARY)

# 将 actual_duration_sec 转换为毫秒，如果是 N/A 保留为 None
def duration_to_ms(x):
    try:
        return float(x) * 1000
    except:
        return None

timeout_df['actual_duration_ms'] = timeout_df['actual_duration_sec'].apply(duration_to_ms)

# 按文件名分组，方便一次处理一个 CSV
timeout_groups = timeout_df.groupby('file_name')

# -----------------------
# 遍历每个原始 CSV 文件
# -----------------------
for file_name in os.listdir(CSV_DIR):
    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(CSV_DIR, file_name)
    df = pd.read_csv(file_path)

    # 检查是否有对应的超时任务
    if file_name in timeout_groups.groups:
        group = timeout_groups.get_group(file_name)
        # 对每条超时任务进行匹配和修改
        for idx, trow in group.iterrows():
            start_time = trow['start_time_ms']
            actual_ms = trow['actual_duration_ms']

            # 找到原始 CSV 中 start_time_ms 相同的行
            mask = df['start_time_ms'] == start_time
            if mask.any():
                if actual_ms is not None:
                    df.loc[mask, 'end_time_ms'] = df.loc[mask, 'start_time_ms'] + actual_ms
                else:
                    df.loc[mask, 'end_time_ms'] = "N/A"
            else:
                print(f"⚠️ 在 {file_name} 中找不到 start_time_ms = {start_time} 对应的行")

    # 保存修正后的 CSV
    out_path = os.path.join(OUTPUT_DIR, file_name)
    df.to_csv(out_path, index=False)
    print(f"✅ 修正完成: {file_name} → {out_path}")
