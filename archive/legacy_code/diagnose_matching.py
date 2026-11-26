"""
诊断脚本：定位 Test_dataset.csv 与 Baseline 文件匹配失败的原因
"""

import csv
from pathlib import Path

# 路径配置
SCRIPT_DIR = Path(__file__).parent
TEST_DATASET_PATH = SCRIPT_DIR / "Test_dataset.csv"
BASELINE_DIR = SCRIPT_DIR.parent / "Baseline"

# Step1 基线文件（qwen3_5_4b）
BASELINE_QWEN35_4B_STEP1 = BASELINE_DIR / "Step1" / "test_qwen3_5_4b-q8_0_False_20260410_180517_annotated.csv"


def read_csv_test_ids(csv_path: Path) -> set:
    """读取 CSV 文件中的所有 test_id"""
    test_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = row.get('test_id', '').strip()
            if test_id:
                test_ids.add(test_id)
    return test_ids


def read_csv_by_step(csv_path: Path, target_step: str) -> list:
    """读取 CSV 文件中指定 step 的行"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_step = row.get('source_step', '').strip()
            if source_step == target_step:
                rows.append(row)
    return rows


def main():
    print("=" * 80)
    print("诊断 Test_dataset.csv 与 Baseline 文件匹配问题")
    print("=" * 80)
    
    # 1. 读取 Test_dataset.csv
    print("\n【1】读取 Test_dataset.csv")
    test_rows_step1 = read_csv_by_step(TEST_DATASET_PATH, "Step1")
    test_rows_step2 = read_csv_by_step(TEST_DATASET_PATH, "Step2")
    test_rows_step3 = read_csv_by_step(TEST_DATASET_PATH, "Step3")
    
    print(f"   Step1: {len(test_rows_step1)} 行")
    print(f"   Step2: {len(test_rows_step2)} 行")
    print(f"   Step3: {len(test_rows_step3)} 行")
    print(f"   总计: {len(test_rows_step1) + len(test_rows_step2) + len(test_rows_step3)} 行")
    
    # 2. 读取基线文件
    print("\n【2】读取基线文件")
    baseline_test_ids = read_csv_test_ids(BASELINE_QWEN35_4B_STEP1)
    print(f"   Baseline qwen3_5_4b Step1: {len(baseline_test_ids)} 个唯一 test_id")
    
    # 3. 提取 Test_dataset Step1 的 test_id
    print("\n【3】Test_dataset Step1 的 test_id")
    test_ids_step1 = set()
    for row in test_rows_step1:
        test_id = row['test_id'].strip()
        test_ids_step1.add(test_id)
    print(f"   唯一 test_id 数量: {len(test_ids_step1)}")
    
    # 4. 计算交集和差集
    print("\n【4】匹配分析")
    intersection = test_ids_step1 & baseline_test_ids
    in_test_not_baseline = test_ids_step1 - baseline_test_ids
    in_baseline_not_test = baseline_test_ids - test_ids_step1
    
    print(f"   ✅ 匹配的 test_id: {len(intersection)}")
    print(f"   ❌ Test_dataset 中有但基线中没有: {len(in_test_not_baseline)}")
    print(f"   📦 基线中有但 Test_dataset 中没有: {len(in_baseline_not_test)}")
    
    # 5. 列出不匹配的 test_id
    if in_test_not_baseline:
        print("\n【5】Test_dataset 中有但基线中**没有**的 test_id:")
        for tid in sorted(in_test_not_baseline):
            # 查找对应的行
            for row in test_rows_step1:
                if row['test_id'].strip() == tid:
                    prompt_preview = row.get('prompt', '')[:50]
                    print(f"   - {tid} (prompt: {prompt_preview}...)")
                    break
    
    # 6. 检查这些缺失的 test_id 是否在其他基线文件中
    if in_test_not_baseline:
        print("\n【6】检查缺失的 test_id 是否在其他基线文件中")
        other_baseline_files = [
            BASELINE_DIR / "Step1" / "test_qwen3_5_9b-q8_0_False_20260410_113731_annotated.csv",
            BASELINE_DIR / "Step1" / "test_qwen3_5_35b-a3b-q8_0_False_20260409_235358_annotated.csv",
            BASELINE_DIR / "Step1" / "test_glm-4_7-flash_q8_0_False_20260412_011535_annotated.csv",
        ]
        
        for other_file in other_baseline_files:
            if other_file.exists():
                other_ids = read_csv_test_ids(other_file)
                found_in_other = in_test_not_baseline & other_ids
                if found_in_other:
                    print(f"   ✓ {other_file.name}:")
                    for tid in sorted(found_in_other):
                        print(f"     - {tid}")
    
    # 7. 详细检查每个 test_id 的格式
    print("\n【7】检查 test_id 格式")
    print("   Test_dataset Step1 的 test_id (带引号显示):")
    for row in test_rows_step1[:5]:
        test_id = row['test_id']
        print(f"     repr: {repr(test_id)}")
    
    print("   Baseline 的 test_id (带引号显示):")
    with open(BASELINE_QWEN35_4B_STEP1, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 5:
                break
            test_id = row['test_id']
            print(f"     repr: {repr(test_id)}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    main()