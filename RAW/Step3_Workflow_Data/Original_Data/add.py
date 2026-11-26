import os
import pandas as pd

# -----------------------------
# 配置
# -----------------------------
ANNOTATED_DIR = "reports2_annotation"    # 手动标注的文件夹
FIXED_TIMESTAMP_DIR = "reports2_fixed"   # 修复时间戳的文件夹
OUTPUT_DIR = "final_merged2"            # 合并后的文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 遍历文件
# -----------------------------
for file_name in os.listdir(ANNOTATED_DIR):
    if not file_name.endswith(".csv"):
        continue

    annotated_path = os.path.join(ANNOTATED_DIR, file_name)
    fixed_path = os.path.join(FIXED_TIMESTAMP_DIR, file_name)

    # 修复文件不存在就跳过
    if not os.path.exists(fixed_path):
        print(f"⚠️ 修复文件不存在: {file_name}，跳过")
        continue

    # 读取两个文件
    df_anno = pd.read_csv(annotated_path)
    df_fixed = pd.read_csv(fixed_path)

    df_anno = pd.read_csv(annotated_path).dropna(how="all")
    df_fixed = pd.read_csv(fixed_path).dropna(how="all")


    # 校验修复文件必须包含需要覆盖的列
    required_cols = {"start_time_ms", "end_time_ms"}
    if not required_cols.issubset(df_fixed.columns):
        print(f"❌ 修复文件缺少必要列: {file_name}，跳过")
        continue

    # 长度一致才能逐行覆盖
    if len(df_anno) != len(df_fixed):
        print(f"⚠️ 行数不一致: {file_name} (标注 {len(df_anno)} vs 修复 {len(df_fixed)})，尝试按 row index 覆盖")
    
    # 直接按 row index 覆盖
    df_anno["start_time_ms"] = df_fixed["start_time_ms"].values
    df_anno["end_time_ms"] = df_fixed["end_time_ms"].values

    # 保存
    out_path = os.path.join(OUTPUT_DIR, file_name)
    df_anno.to_csv(out_path, index=False)
    print(f"✅ 已覆盖并保存: {out_path}")
