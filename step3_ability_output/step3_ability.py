#!/usr/bin/env python3
"""
step3_ability.py — Ability Score Calculation (per-judge, no cross-judge aggregation)

Usage:
    cd <project_root>
    python step3_ability.py

Output: step3_ability_output/
"""

import os, sys, re, warnings, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

# ============================================================
# CONFIG — 所有常量集中于此
# ============================================================
CONFIG = {
    "data_root": "step2.2_vs_baseline",
    "out_dir": "step3_ability_output",

    "cols": {
        "item_id": "test_id",
        "vs_1": "vs_1",
        "vs_2": "vs_2",
    },

    # 文件名中的 baseline → anchor 映射
    "baseline_to_anchor": {
        "qwen3_5_4b": "weak",
        "qwen3_5_9b": "mid",
        "qwen3_5_35b": "strong",
        "glm4_7_flash": "cross",
    },

    # Step1 子 step 划分
    "substep_split": {
        "Step1": [(1, 100, "ArenaHard"), (101, 150, "LiveCodeBench"), (151, 200, "GPQA")],
    },

    "W_LOSS": 0.0,
    "W_BIG_LOSS": 0.0,

    # item 内双向合并是否复刻校准的"取整到整数 ordinal"
    "merge_rounding": False,

    "invalid_tokens": ["error", "ERROR", "", "nan", "NaN", "error\n", "Error"],

    "anchors": ["weak", "mid", "strong", "cross"],

    # benchmark scores for D_k computation
    "benchmark_scores": {
        "weak":   (76.2, 55.8, 79.9),
        "mid":    (81.7, 65.6, 79.1),
        "strong": (84.2, 74.6, 81.2),
        "cross":  (75.2, 64.0, 79.5),
    },
}

# ============================================================
# Constants derived from CONFIG
# ============================================================
VALID_LABELS = frozenset({"A>>B", "A>B", "A=B", "A<B", "A<<B"})

W_TABLE = {
    "A>>B": 3.0,
    "A>B":  1.0,
    "A=B":  0.5,
    "A<B":  CONFIG["W_LOSS"],
    "A<<B": CONFIG["W_BIG_LOSS"],
}

ORDINAL = {"A<<B": 1, "A<B": 2, "A=B": 3, "A>B": 4, "A>>B": 5}
ORD_W = {1: CONFIG["W_BIG_LOSS"], 2: CONFIG["W_LOSS"], 3: 0.5, 4: 1.0, 5: 3.0}

# vs_2 反转：Baseline 视角 ↔ Test 视角
REVERSE_LABEL = {
    "A>>B": "A<<B", "A>B": "A<B", "A=B": "A=B",
    "A<B": "A>B", "A<<B": "A>>B",
}

# 算子匹配顺序（长度降序，避免 >> 被 > 抢匹配）
OPS = [">>", "<<", "=", ">", "<"]

# normalize 状态常量
STATUS_VALID = "valid"
STATUS_INVALID_KNOWN = "invalid_known"
STATUS_UNKNOWN = "unknown"

# ============================================================
# normalize_label — 算子驱动重建，三类返回
# ============================================================
def normalize_label(raw):
    """
    从可能截断的标签文本重建标准标签。

    Returns:
        (label_or_None, status)
        status ∈ {STATUS_VALID, STATUS_INVALID_KNOWN, STATUS_UNKNOWN}
    """
    raw = str(raw).strip().strip('"').strip("'").strip()

    # 1. 已知无效
    if raw in CONFIG["invalid_tokens"] or raw == "":
        return None, STATUS_INVALID_KNOWN

    # 2. 已是标准标签
    if raw in VALID_LABELS:
        return raw, STATUS_VALID

    # 3. B-first 格式直接映射（B>>A → A<<B 等）
    b_first_map = {"B>>A": "A<<B", "B>A": "A<B", "B=A": "A=B", "B<A": "A>B", "B<<A": "A>>B"}
    if raw in b_first_map:
        return b_first_map[raw], STATUS_VALID

    # 4. 算子驱动重建
    op = None
    op_pos = len(raw) + 1
    for candidate in OPS:
        p = raw.find(candidate)
        if p != -1 and (op is None or p < op_pos):
            op = candidate
            op_pos = p

    if op is None:
        return None, STATUS_UNKNOWN

    left = raw[:op_pos].strip()
    right = raw[op_pos + len(op):].strip()

    # 补缺失侧
    if left == "":
        if right in ("A", "B"):
            left = "A"
        else:
            return None, STATUS_UNKNOWN

    if right == "":
        if left in ("A", "B"):
            right = "B" if left == "A" else "A"
        else:
            return None, STATUS_UNKNOWN

    left = left[0]
    right = right[0]

    if left not in ("A", "B") or right not in ("A", "B"):
        return None, STATUS_UNKNOWN

    reconstructed = f"{left}{op}{right}"

    # B-first 翻转
    if left == "B":
        flip = {">>": "<<", "<<": ">>", ">": "<", "<": ">", "=": "="}
        flipped_op = flip.get(op)
        if flipped_op is None:
            return None, STATUS_UNKNOWN
        reconstructed = f"A{flipped_op}B"

    if reconstructed in VALID_LABELS:
        return reconstructed, STATUS_VALID

    return None, STATUS_UNKNOWN


# ============================================================
# reverse_label — vs_2 反转
# ============================================================
def reverse_label(label):
    """将 label 从 Baseline 视角翻转为 Test 视角。"""
    return REVERSE_LABEL.get(label)


# ============================================================
# resolve_substep
# ============================================================
def resolve_substep(test_step, item_id):
    """
    根据 test_step 和 item_id 解析子 step。
    Step1 按区间拆分；非 Step1 原样返回。
    """
    if test_step != "Step1":
        return test_step

    m = re.search(r'(\d+)$', str(item_id))
    if m is None:
        return "Step1_NODIGITS"

    item_num = int(m.group(1)) + 1  # test_0000 → item 1

    if test_step in CONFIG["substep_split"]:
        for start, end, substep_name in CONFIG["substep_split"][test_step]:
            if start <= item_num <= end:
                return substep_name

    return f"Step1_UNMAPPED_{item_num}"


# ============================================================
# load_data — 从目录结构发现并加载所有 CSV
# ============================================================
def load_data(cfg):
    """
    扫描 data_root/{Step}/{judge}/*.csv, 构建统一 DataFrame。
    排除 .sync-conflict-* 文件。
    """
    data_root = Path(cfg["data_root"])
    if not data_root.is_dir():
        print(f"[FATAL] data_root '{data_root}' not found")
        sys.exit(1)

    rows = []
    csv_count = 0
    error_files = 0

    # 查找所有 Step 目录
    for step_dir in sorted(data_root.iterdir()):
        if not step_dir.is_dir():
            continue
        test_step = step_dir.name

        # 遍历 judge 目录
        for judge_dir in sorted(step_dir.iterdir()):
            if not judge_dir.is_dir():
                continue
            judge_model = judge_dir.name

            # 遍历 CSV
            for fpath in sorted(judge_dir.glob("*.csv")):
                if ".sync-conflict" in fpath.name:
                    continue

                csv_count += 1
                try:
                    df = pd.read_csv(fpath, dtype=str, keep_default_na=False)
                except Exception as e:
                    print(f"  [WARN] Failed to read {fpath.name}: {e}")
                    error_files += 1
                    continue

                if df.empty:
                    continue

                # 从 filename 提取 test_model 和 baseline
                stem = fpath.stem  # without .csv
                parts = stem.split("_vs_")
                if len(parts) != 2:
                    print(f"  [WARN] Unexpected filename pattern: {fpath.name}, skip")
                    error_files += 1
                    continue

                test_model = parts[0]
                # Normalize: hyphens→underscores, _True/_False→_true/_false
                test_model = test_model.replace("-", "_")
                test_model = re.sub(r'_(true|false)(_|$)', lambda m: '_' + m.group(1).lower() + m.group(2),
                                    test_model, flags=re.IGNORECASE)
                test_model = re.sub(r'_(true|false)$', lambda m: '_' + m.group(1).lower(),
                                    test_model, flags=re.IGNORECASE)
                baseline_raw = parts[1]
                if 'deepseek_moe' in test_model.lower() or not re.search(r'_reports[12]$', test_model):
                    continue
                baseline_anchor = cfg["baseline_to_anchor"].get(baseline_raw)
                if baseline_anchor is None:
                    print(f"  [WARN] Unknown baseline '{baseline_raw}' in {fpath.name}, skip")
                    error_files += 1
                    continue

                # Match saved judgements to the exact Dataset/Baseline question.
                if test_step == "Step3":
                    from pipeline_sources import select_workflow_rows
                    df = select_workflow_rows(df, test_model, baseline_raw)
                else:
                    id_col = cfg["cols"]["item_id"]
                    if id_col not in df.columns:
                        print(f"  [WARN] Column '{id_col}' not found in {fpath.name}, skip")
                        error_files += 1
                        continue
                    df["_item_id"] = df[id_col].astype(str).str.strip()

                df["_test_model"] = test_model
                df["_test_step"] = test_step
                df["_baseline_anchor"] = baseline_anchor
                df["_judge_model"] = judge_model

                # 保留 vs_1, vs_2 原始值
                rows.append(df[["_test_model", "_test_step", "_item_id",
                                "_baseline_anchor", "_judge_model",
                                cfg["cols"]["vs_1"], cfg["cols"]["vs_2"]]].copy())

    if not rows:
        print(f"[FATAL] No data loaded. Check data_root path.")
        sys.exit(1)

    result = pd.concat(rows, ignore_index=True)
    result.columns = ["test_model", "test_step", "item_id",
                      "baseline_anchor", "judge_model", "vs_1", "vs_2"]

    print(f"\nLoaded {len(result)} rows from {csv_count} CSVs ({error_files} errors)")
    return result


# ============================================================
# validate_and_normalize — 对 vs_1 / vs_2 做标签校验与重建
# ============================================================
def validate_and_normalize(df):
    """验证并标准化 vs_1, vs_2。返回增强 DataFrame + 统计信息。"""
    df = df.copy()
    stats = {
        "vs1_valid": 0, "vs1_invalid_known": 0, "vs1_unknown": 0,
        "vs2_valid": 0, "vs2_invalid_known": 0, "vs2_unknown": 0,
        "unknown_values": defaultdict(list),  # raw_value → [(idx, col)]
    }

    norm_vs1 = []
    norm_vs2 = []
    status_vs1 = []
    status_vs2 = []

    for i, row in df.iterrows():
        for col, norm_list, status_list, stats_prefix in [
            ("vs_1", norm_vs1, status_vs1, "vs1"),
            ("vs_2", norm_vs2, status_vs2, "vs2"),
        ]:
            raw = row[col]
            label, st = normalize_label(raw)
            norm_list.append(label)
            status_list.append(st)
            stats[f"{stats_prefix}_{st}"] += 1
            if st == STATUS_UNKNOWN:
                stats["unknown_values"][str(raw)].append((i, col))

    df["vs_1_norm"] = norm_vs1
    df["vs_2_norm"] = norm_vs2
    df["vs_1_status"] = status_vs1
    df["vs_2_status"] = status_vs2

    return df, stats


# ============================================================
# resolve_substeps — 应用到全表
# ============================================================
def apply_resolve_substep(df):
    """为每一行计算 substep。"""
    df = df.copy()
    df["substep"] = df.apply(lambda r: resolve_substep(r["test_step"], r["item_id"]), axis=1)

    unmapped = df["substep"].str.startswith("Step1_UNMAPPED").sum()
    if unmapped:
        print(f"  [WARN] {unmapped} rows in Step1 outside all mapped ranges:")

    return df


# ============================================================
# merge_per_item — 双向合并
# ============================================================
def merge_per_item(df, cfg):
    """
    按 item 合并 vs_1 和 vs_2 为单一 item_W。
    vs_2 先 reverse 到 Test 视角再查 W_TABLE。
    """
    df = df.copy()
    merge_rounding = cfg["merge_rounding"]

    item_w_list = []
    n_dirs_list = []
    w_vs1_list = []
    w_vs2_list = []

    total_both = 0
    total_only_vs1 = 0
    total_only_vs2 = 0
    total_both_invalid = 0

    for _, row in df.iterrows():
        # vs_1
        vs1_ok = row["vs_1_status"] == STATUS_VALID
        if vs1_ok:
            w1 = W_TABLE[row["vs_1_norm"]]
            ord1 = ORDINAL[row["vs_1_norm"]]
        else:
            w1 = None

        # vs_2: 先 reverse 到 Test 视角
        vs2_ok = row["vs_2_status"] == STATUS_VALID
        if vs2_ok:
            vs2_test = reverse_label(row["vs_2_norm"])
            # reverse_label should always return valid for valid input
            w2 = W_TABLE[vs2_test]
            ord2 = ORDINAL[vs2_test]
        else:
            w2 = None

        # 合并
        if vs1_ok and vs2_ok:
            total_both += 1
            if merge_rounding:
                avg_ord = math.floor((ord1 + ord2) / 2.0 + 0.5)
                item_w = ORD_W[avg_ord]
            else:
                item_w = np.mean([w1, w2])
            n_dirs = 2
        elif vs1_ok and not vs2_ok:
            total_only_vs1 += 1
            item_w = w1
            n_dirs = 1
        elif not vs1_ok and vs2_ok:
            total_only_vs2 += 1
            item_w = w2
            n_dirs = 1
        else:
            total_both_invalid += 1
            item_w = np.nan
            n_dirs = 0

        item_w_list.append(item_w)
        n_dirs_list.append(n_dirs)
        w_vs1_list.append(w1)
        w_vs2_list.append(w2)

    df["w_vs1"] = w_vs1_list
    df["w_vs2"] = w_vs2_list
    df["n_dirs"] = n_dirs_list
    df["item_W"] = item_w_list

    # 丢弃两向无效的行
    dropped_mask = df["item_W"].isna()
    n_dropped = dropped_mask.sum()
    if n_dropped > 0:
        print(f"  [INFO] Dropping {n_dropped} items with both directions invalid")
        df = df[~dropped_mask].copy()

    merge_stats = {
        "both_valid": total_both,
        "only_vs1": total_only_vs1,
        "only_vs2": total_only_vs2,
        "both_invalid": total_both_invalid,
        "dropped": n_dropped,
    }

    return df, merge_stats


# ============================================================
# compute_D — 从 benchmark 表计算难度系数
# ============================================================
def compute_D(cfg):
    """
    复合能力指数 = (GPQA + LCB_v6 + τ²-Bench)/3
    D_k = avg_k / avg_weak
    全精度，不硬编码。
    """
    scores = cfg["benchmark_scores"]
    avgs = {}
    for anchor, (gpqa, lcb, tau) in scores.items():
        avgs[anchor] = (gpqa + lcb + tau) / 3.0

    avg_weak = avgs["weak"]
    D = {}
    for anchor in cfg["anchors"]:
        D[anchor] = avgs[anchor] / avg_weak

    return D


# ============================================================
# aggregate_W — 对每个 (test_model, step, baseline_anchor, judge) 取 item_W 均值
# ============================================================
def aggregate_W(df):
    """W_{A,k,n,step} = mean(item_W) per group."""
    gb = df.groupby(["test_model", "test_step", "baseline_anchor", "judge_model", "substep"],
                    observed=False)
    result = gb["item_W"].agg(["mean", "count"]).reset_index()
    result.columns = ["test_model", "test_step", "baseline_anchor",
                      "judge_model", "substep", "W", "n_items"]
    return result


# ============================================================
# compute_ability
# ============================================================
def compute_ability(W_df, D):
    """
    对每个 (test_model, step, judge_model)：
    Ability = Σ_{k ∈ anchors} (W_{A,k,n,step} × D_k)

    缺 anchor 时报告，不补 0。
    """
    anchors = CONFIG["anchors"]
    records = []

    groups = W_df.groupby(["test_model", "test_step", "judge_model", "substep"],
                          observed=False)

    for (tm, ts, jm, substep), grp in groups:
        w_map = dict(zip(grp["baseline_anchor"], grp["W"]))
        missing = [a for a in anchors if a not in w_map]
        if missing:
            print(f"  [WARN] ({tm}, {ts}, {jm}, {substep}): missing anchors {missing}")

        if missing:
            ability = float("nan")
        else:
            ability = sum(w_map[k] * D[k] for k in anchors)
        records.append({
            "test_model": tm,
            "test_step": ts,
            "judge_model": jm,
            "substep": substep,
            "Ability": ability,
            "n_anchors_present": len(anchors) - len(missing),
            "missing_anchors": ",".join(missing) if missing else "",
        })

    return pd.DataFrame(records)


# ============================================================
# save_all — 输出 5 个文件
# ============================================================
def save_all(D, item_merged, W_scores, ability, cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. D_coefficients.csv
    d_records = []
    for anchor in cfg["anchors"]:
        gpqa, lcb, tau = cfg["benchmark_scores"][anchor]
        avg = (gpqa + lcb + tau) / 3.0
        d_records.append({"anchor": anchor, "avg": avg, "D_k": D[anchor]})
    pd.DataFrame(d_records).to_csv(out_dir / "D_coefficients.csv", index=False, float_format="%.17g")
    print(f"  Saved D_coefficients.csv")

    # 2. item_merged.csv
    out_cols = ["test_model", "test_step", "baseline_anchor", "judge_model",
                "item_id", "substep", "w_vs1", "w_vs2", "n_dirs", "item_W"]
    item_merged[out_cols].to_csv(out_dir / "item_merged.csv", index=False, float_format="%.17g")
    print(f"  Saved item_merged.csv ({len(item_merged)} rows)")

    # 3. W_scores.csv
    W_scores.to_csv(out_dir / "W_scores.csv", index=False, float_format="%.17g")
    print(f"  Saved W_scores.csv ({len(W_scores)} rows)")

    # 4. ability_scores.csv (long)
    ability.to_csv(out_dir / "ability_scores.csv", index=False, float_format="%.17g")
    print(f"  Saved ability_scores.csv ({len(ability)} rows)")

    # 5. ability_wide.csv
    # 宽表：index=(test_model, test_step, substep), columns=judge_model
    ability_pivot = ability.pivot_table(
        index=["test_model", "test_step", "substep"],
        columns="judge_model",
        values="Ability",
        aggfunc="first",
    )
    ability_pivot.to_csv(out_dir / "ability_wide.csv", float_format="%.17g")
    print(f"  Saved ability_wide.csv ({len(ability_pivot)} rows)")


# ============================================================
# self_check — 全面自检
# ============================================================
def self_check(D, item_merged, W_scores, ability, label_stats, merge_stats, cfg):
    print("\n" + "=" * 60)
    print("  SELF CHECK")
    print("=" * 60)

    all_ok = True

    # ---- 1. D_k 精度 ----
    print("\n--- D_k precision check ---")
    expected = {"weak": 1.0000, "mid": 1.0684, "strong": 1.1326, "cross": 1.0321}
    for anchor in cfg["anchors"]:
        dk = D[anchor]
        exp = expected[anchor]
        diff = abs(dk - exp)
        ok = diff < 1e-3
        status = "OK" if ok else "FAIL"
        print(f"  D_{anchor} = {dk:.6f}  (expected ~{exp:.4f})  diff={diff:.6f}  [{status}]")
        if not ok:
            all_ok = False

    # ---- 2. Substep item 去重计数 ----
    print("\n--- Step1 substep unique items ---")
    step1 = item_merged[item_merged["test_step"] == "Step1"]
    expected_counts = {"ArenaHard": 100, "LiveCodeBench": 50, "GPQA": 50}
    for sname, expected_count in expected_counts.items():
        actual = step1[step1["substep"] == sname]["item_id"].nunique()
        ok = actual == expected_count
        status = "OK" if ok else "WARN"
        print(f"  {sname}: {actual}  (expected {expected_count})  [{status}]")
        if not ok:
            all_ok = False

    unmapped = step1[step1["substep"].str.startswith("Step1_UNMAPPED", na=False)]
    if len(unmapped) > 0:
        print(f"  UNMAPPED Step1 items: {len(unmapped)}  [WARN]")
        print(f"    Examples: {unmapped['item_id'].unique()[:5]}")
        all_ok = False
    else:
        print(f"  UNMAPPED: 0  [OK]")

    # ---- 3. 双向覆盖统计 ----
    print("\n--- Bidirectional coverage ---")
    ms = merge_stats
    total_merge = ms["both_valid"] + ms["only_vs1"] + ms["only_vs2"] + ms["both_invalid"]
    print(f"  Both directions valid:  {ms['both_valid']:>6} ({ms['both_valid']/total_merge*100:.1f}%)")
    print(f"  Only vs_1 valid:       {ms['only_vs1']:>6} ({ms['only_vs1']/total_merge*100:.1f}%)")
    print(f"  Only vs_2 valid:       {ms['only_vs2']:>6} ({ms['only_vs2']/total_merge*100:.1f}%)")
    print(f"  Both invalid (dropped): {ms['both_invalid']:>6} ({ms['both_invalid']/total_merge*100:.1f}%)")
    print(f"  Dropped from merge:    {ms['dropped']:>6}")

    # 单向 vs_1 / vs_2 invalid from label normalization
    print(f"\n  vs_1 invalid (known):   {label_stats['vs1_invalid_known']:>6}")
    print(f"  vs_1 unknown:           {label_stats['vs1_unknown']:>6}")
    print(f"  vs_2 invalid (known):   {label_stats['vs2_invalid_known']:>6}")
    print(f"  vs_2 unknown:           {label_stats['vs2_unknown']:>6}")

    # ---- 4. 单向比例警告 ----
    print("\n--- Single-direction ratio per (model, step, baseline, judge) ---")
    single_warn_count = 0
    for (tm, ts, ba, jm), grp in W_scores.groupby(
            ["test_model", "test_step", "baseline_anchor", "judge_model"], observed=False):
        # need to look at item_merged for the per-item n_dirs
        sub = item_merged[
            (item_merged["test_model"] == tm) &
            (item_merged["test_step"] == ts) &
            (item_merged["baseline_anchor"] == ba) &
            (item_merged["judge_model"] == jm)
        ]
        if len(sub) == 0:
            continue
        n_total = len(sub)
        n_single = (sub["n_dirs"] == 1).sum()
        ratio = n_single / n_total
        if ratio > 0.5:
            print(f"  [WARN] {tm} | {ts} | {ba} | {jm}: "
                  f"{n_single}/{n_total} items are single-direction ({ratio:.1%})")
            single_warn_count += 1
    if single_warn_count == 0:
        print(f"  No groups with >50% single-direction items  [OK]")

    # ---- 5. Anchor 覆盖 ----
    print("\n--- Anchor coverage per (test_model, step, judge) ---")
    anchor_coverage = W_scores.groupby(
        ["test_model", "test_step", "judge_model", "substep"], observed=False
    )["baseline_anchor"].apply(set).reset_index()

    incomplete = []
    for _, row in anchor_coverage.iterrows():
        missing = set(cfg["anchors"]) - row["baseline_anchor"]
        if missing:
            incomplete.append((row["test_model"], row["test_step"],
                               row["judge_model"], row["substep"], missing))

    if incomplete:
        print(f"  {len(incomplete)} groups have incomplete anchor coverage:")
        for tm, ts, jm, substep, missing in incomplete[:10]:
            print(f"    ({tm}, {ts}, {jm}, {substep}): missing {missing}")
        if len(incomplete) > 10:
            print(f"    ... and {len(incomplete) - 10} more")
        all_ok = False
    else:
        print(f"  All groups have all 4 anchors  [OK]")

    # ---- 6. Unknown 标签报告 ----
    print("\n--- Unknown label values ---")
    unknown_vals = label_stats.get("unknown_values", {})
    if unknown_vals:
        print(f"  Found {sum(len(v) for v in unknown_vals.values())} unknown label(s):")
        for raw_val, occurrences in sorted(unknown_vals.items(), key=lambda x: -len(x[1])):
            count_vs1 = sum(1 for _, col in occurrences if col == "vs_1")
            count_vs2 = sum(1 for _, col in occurrences if col == "vs_2")
            print(f"    {repr(raw_val)}: {len(occurrences)} occurrences "
                  f"(vs_1={count_vs1}, vs_2={count_vs2})")
            for idx, col in occurrences[:3]:
                print(f"      row={idx}, col={col}")
        all_ok = False
    else:
        print(f"  No unknown labels found  [OK]")

    # ---- 7. NaN / 丢弃汇总 ----
    print("\n--- Data integrity summary ---")
    print(f"  Total rows loaded:         {label_stats.get('total_rows', '?')}")
    print(f"  Rows after merge drop:     {len(item_merged)}")
    print(f"  NaN in item_W (should be 0): {item_merged['item_W'].isna().sum()}")
    if item_merged["item_W"].isna().sum() > 0:
        print(f"  [WARN] NaN values remain in item_W!")
        all_ok = False
    print(f"  NaN in W_scores:           {W_scores['W'].isna().sum()}")
    print(f"  NaN in ability:            {ability['Ability'].isna().sum()}")

    # ---- Final ----
    print(f"\n{'=' * 60}")
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED - review warnings above")
    print("=" * 60)


# ============================================================
# main
# ============================================================
def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    print("=" * 60)
    print("  ABILITY SCORE CALCULATION (Step 3)")
    print("=" * 60)

    cfg = CONFIG

    # 1. Load
    print("\n--- Loading data ---")
    df = load_data(cfg)
    label_stats = {"total_rows": len(df)}

    # 2. Validate & normalize labels
    print("\n--- Validating labels ---")
    df, ls = validate_and_normalize(df)
    for k, v in ls.items():
        if k != "unknown_values":
            label_stats[k] = v
    label_stats["unknown_values"] = ls["unknown_values"]
    print(f"  vs_1: valid={ls['vs1_valid']}, invalid_known={ls['vs1_invalid_known']}, "
          f"unknown={ls['vs1_unknown']}")
    print(f"  vs_2: valid={ls['vs2_valid']}, invalid_known={ls['vs2_invalid_known']}, "
          f"unknown={ls['vs2_unknown']}")

    # 3. Resolve substep
    print("\n--- Resolving substeps ---")
    df = apply_resolve_substep(df)
    # Preserve all selected questions, including records whose two labels are
    # unusable; the existing averaging rule excludes those labels explicitly.
    Path(cfg['out_dir']).mkdir(parents=True, exist_ok=True)
    df[df.test_step.eq('Step3')].to_csv(Path(cfg['out_dir'])/'workflow_selected_items.csv', index=False)
    omitted = df[df.vs_1_status.ne(STATUS_VALID) & df.vs_2_status.ne(STATUS_VALID)]
    omitted.to_csv(Path(cfg['out_dir'])/'unusable_judgement_items.csv', index=False)

    # 4. Merge per item (bidirectional)
    print("\n--- Merging per item ---")
    df, merge_stats = merge_per_item(df, cfg)

    # 5. Compute D_k
    print("\n--- Computing D_k ---")
    D = compute_D(cfg)

    # 6. Aggregate W
    print("\n--- Aggregating W scores ---")
    W_scores = aggregate_W(df)
    print(f"  {len(W_scores)} groups")

    # 7. Compute Ability
    print("\n--- Computing Ability scores ---")
    ability = compute_ability(W_scores, D)
    print(f"  {len(ability)} (test_model, step, judge) entries")

    # 8. Save
    print("\n--- Saving outputs ---")
    save_all(D, df, W_scores, ability, cfg)

    # 9. Self-check
    self_check(D, df, W_scores, ability, label_stats, merge_stats, cfg)


if __name__ == "__main__":
    main()
