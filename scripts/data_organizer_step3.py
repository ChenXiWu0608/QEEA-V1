#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据整理脚本 - Step3 (合并版)
功能：
1. 从JSON文件加载映射数据（包含category, subcategory, language）
2. 应用硬编码的映射规则进行转换
3. 遍历Baseline/Step3目录下的所有CSV文件
4. 为每条数据追加category、subcategory、language三个字段
5. 删除原始的workflow_name和question_index列
6. 保存为_annotated.csv文件

Mapping Rules (based on workflow_name and question_index):
- customer_bot + 1 -> category=customer_support, subcategory=workflow_cs_simple
- customer_bot + 2 -> category=customer_support, subcategory=workflow_cs_conflict
- customer_bot + 3 -> category=customer_support, subcategory=workflow_cs_prediction
- deep_research + 1-3 -> category=research, subcategory=investigation
- multilingual_translation + 1-2 -> category=translation, subcategory=multilingual
"""

import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# 硬编码的映射规则
# 格式: (workflow_name, question_index) -> (new_category, new_subcategory)
MAPPING_RULES = {
    ('customer_bot', 1): ('customer_support', 'workflow_cs_simple'),
    ('customer_bot', 2): ('customer_support', 'workflow_cs_conflict'),
    ('customer_bot', 3): ('customer_support', 'workflow_cs_prediction'),
    ('deep_research', 1): ('research', 'investigation'),
    ('deep_research', 2): ('research', 'investigation'),
    ('deep_research', 3): ('research', 'investigation'),
    ('multilingual_translation', 1): ('translation', 'multilingual'),
    ('multilingual_translation', 2): ('translation', 'multilingual'),
}


def load_json_data(json_path):
    """加载JSON文件，返回列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def clean_text(text):
    """清理文本中的换行符和多余空白"""
    if not text:
        return ''
    # 将换行符替换为空格
    text = text.replace('\n', ' ').replace('\r', ' ')
    # 移除多余空格
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def convert_question_index(qi):
    """转换 question_index 格式，将 '1月1日' 转换为 '1-1'"""
    if isinstance(qi, int):
        return qi
    if isinstance(qi, float):
        if qi == int(qi):
            return int(qi)
        return qi
    if isinstance(qi, str):
        # 尝试直接转换为整数
        try:
            return int(qi)
        except ValueError:
            pass
        # 处理 '1月1日' 格式 -> '1-1'
        if '月' in qi and '日' in qi:
            qi = qi.replace('月', '-').replace('日', '')
            return qi
    return qi


def build_step3_map(json_data):
    """构建Step3的索引映射"""
    # Step3 JSON 数据结构 (新格式):
    # [
    #   {"uid": "...", "workflow_name": "customer_bot", "question_index": 1, "category": "...", "subcategory": "...", "language": "..."},
    #   ...
    # ]
    # 按 workflow_name + question_index 构建映射
    grouped_data = {}
    for item in json_data:
        workflow_name = item['workflow_name']
        question_index = item['question_index']
        
        key = (workflow_name, question_index)
        grouped_data[key] = {
            'category': clean_text(item.get('category', '')),
            'subcategory': clean_text(item.get('subcategory', '')),
            'language': clean_text(item.get('language', ''))
        }
    
    return grouped_data


def apply_mapping_to_value(workflow_name, question_index, step3_map=None):
    """
    将workflow_name和question_index转换为category和subcategory
    
    Args:
        workflow_name: 原始workflow_name值（标量字符串）
        question_index: 原始question_index值（标量整数）
        step3_map: 从JSON文件加载的映射（可选）
    
    Returns:
        (new_category, new_subcategory, language) 元组
    """
    # 确保workflow_name是字符串
    if workflow_name is None or (isinstance(workflow_name, float) and pd.isna(workflow_name)):
        workflow_name_str = ''
    else:
        workflow_name_str = str(workflow_name).strip()
    
    # 处理question_index - 转换为整数
    if question_index is None or (isinstance(question_index, float) and pd.isna(question_index)):
        question_index_int = None
    else:
        try:
            question_index_int = int(question_index)
        except (ValueError, TypeError):
            try:
                question_index_int = int(str(question_index).strip())
            except (ValueError, TypeError):
                question_index_int = None
    
    # 首先尝试从JSON映射中获取（包含language信息）
    if step3_map:
        # 尝试转换后的键
        converted_qi = convert_question_index(question_index) if question_index_int is None else question_index_int
        key = (workflow_name_str, converted_qi)
        
        if key in step3_map:
            info = step3_map[key]
            return info['category'], info['subcategory'], info['language']
        
        # 尝试原始整数键
        if question_index_int is not None:
            key_int = (workflow_name_str, question_index_int)
            if key_int in step3_map:
                info = step3_map[key_int]
                return info['category'], info['subcategory'], info['language']
    
    # 如果JSON映射中没有，尝试硬编码映射规则
    if question_index_int is not None:
        key = (workflow_name_str, question_index_int)
        if key in MAPPING_RULES:
            new_cat, new_sub = MAPPING_RULES[key]
            return new_cat, new_sub, ''
    
    # 如果没有匹配的规则，返回原始workflow_name和question_index
    return workflow_name_str, str(question_index_int) if question_index_int is not None else '', ''


def process_csv_file(csv_path, output_path, step3_map=None):
    """
    处理单个CSV文件，添加category信息
    
    Args:
        csv_path: 输入CSV文件路径
        output_path: 输出CSV文件路径
        step3_map: 从JSON文件加载的映射
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查是否有workflow_name和question_index列
    if 'workflow_name' not in df.columns or 'question_index' not in df.columns:
        print(f"Skipping (missing workflow_name or question_index column): {csv_path}")
        return None
    
    # 添加新列
    categories = []
    subcategories = []
    languages = []
    original_pairs = []
    mapped_rows = 0
    unchanged_rows = 0
    mappings = defaultdict(int)
    
    for _, row in df.iterrows():
        workflow_name = row['workflow_name']
        question_index = row['question_index']
        
        # 获取标量值
        if workflow_name is None or (isinstance(workflow_name, float) and pd.isna(workflow_name)):
            workflow_name_str = ''
        else:
            workflow_name_str = str(workflow_name).strip()
        
        if question_index is None or (isinstance(question_index, float) and pd.isna(question_index)):
            question_index_str = ''
            question_index_int = None
        else:
            try:
                question_index_int = int(question_index)
                question_index_str = str(question_index_int)
            except (ValueError, TypeError):
                try:
                    question_index_int = int(str(question_index).strip())
                    question_index_str = str(question_index_int)
                except (ValueError, TypeError):
                    question_index_int = None
                    question_index_str = str(question_index).strip() if isinstance(question_index, str) else ''
        
        original_pairs.append((workflow_name_str, question_index_str))
        
        # 应用映射
        new_category, new_subcategory, new_language = apply_mapping_to_value(
            workflow_name_str, question_index_int if question_index_int is not None else question_index_str, step3_map
        )
        categories.append(new_category)
        subcategories.append(new_subcategory)
        languages.append(new_language)
    
    # 创建新列
    df['category'] = categories
    df['subcategory'] = subcategories
    df['language'] = languages
    
    # 统计映射情况
    for orig_workflow, orig_index in original_pairs:
        idx = original_pairs.index((orig_workflow, orig_index))
        new_cat = categories[idx]
        new_sub = subcategories[idx]
        
        # 检查是否进行了映射（与原始workflow_name不同）
        if new_cat != orig_workflow or new_sub != orig_index:
            mapped_rows += 1
            mappings[f"({orig_workflow}, {orig_index}) -> ({new_cat}, {new_sub})"] += 1
        else:
            unchanged_rows += 1
    
    # 删除旧列
    columns_to_drop = []
    if 'workflow_name' in df.columns:
        columns_to_drop.append('workflow_name')
    if 'question_index' in df.columns:
        columns_to_drop.append('question_index')
    # 也删除旧的category/subcategory列（我们是从workflow_name和question_index生成新的）
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存处理后的文件
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Processed: {csv_path} -> {output_path}")
    print(f"  -> Total rows: {len(df)}, Mapped: {mapped_rows}, Unchanged: {unchanged_rows}")
    
    return {
        'total_rows': len(df),
        'mapped_rows': mapped_rows,
        'unchanged_rows': unchanged_rows,
        'mappings': dict(mappings),
    }


def scan_csv_files(base_dir: str) -> list:
    """递归扫描目录下所有CSV文件"""
    csv_files = []
    base_path = Path(base_dir)
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv') and '_annotated' not in file:
                csv_files.append(os.path.join(root, file))
    
    return sorted(csv_files)


def main():
    """主函数"""
    print("=" * 60)
    print("Data Organizer - Step3 (合并版)")
    print("=" * 60)
    print()
    
    # 定义数据源路径
    step3_path = 'Test_file/Step3_Workflow/tasks_converted.json'
    
    # 加载JSON数据
    step3_map = None
    if os.path.exists(step3_path):
        print(f"Loading JSON data from: {step3_path}")
        step3_data = load_json_data(step3_path)
        step3_map = build_step3_map(step3_data)
        print(f"  -> {len(step3_data)} records loaded")
        print(f"  -> {len(step3_map)} unique workflow_name + question_index combinations")
    else:
        print(f"Warning: JSON file not found: {step3_path}")
        print("  -> Will use hardcoded mapping rules only")
    
    print()
    
    # 定义Dataset/Step3目录
    dataset_step3_path = 'Baseline/Step3'
    
    # 扫描CSV文件
    print(f"Scanning directory: {dataset_step3_path}")
    csv_files = scan_csv_files(dataset_step3_path)
    print(f"Found {len(csv_files)} CSV files to process")
    print()
    
    if not csv_files:
        print("No CSV files found to process. Exiting.")
        return
    
    # 处理每个文件
    total_stats = {
        'total_files': 0,
        'total_rows': 0,
        'total_mapped': 0,
        'total_unchanged': 0,
        'all_mappings': defaultdict(int),
    }
    
    for i, csv_path in enumerate(csv_files, 1):
        # 生成输出文件名
        csv_path_obj = Path(csv_path)
        output_file = csv_path_obj.name.replace('.csv', '_annotated.csv')
        output_path = csv_path_obj.parent / output_file
        
        print(f"[{i}/{len(csv_files)}] Processing: {csv_path_obj.name}")
        
        try:
            stats = process_csv_file(csv_path, output_path, step3_map)
            
            if stats:
                total_stats['total_files'] += 1
                total_stats['total_rows'] += stats['total_rows']
                total_stats['total_mapped'] += stats['mapped_rows']
                total_stats['total_unchanged'] += stats['unchanged_rows']
                
                # 合并映射统计
                for mapping, count in stats['mappings'].items():
                    total_stats['all_mappings'][mapping] += count
        
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # 输出汇总统计
    print("=" * 60)
    print("Processing Complete! Summary Statistics")
    print("=" * 60)
    print(f"Files processed: {total_stats['total_files']}")
    print(f"Total rows: {total_stats['total_rows']}")
    print(f"Mapped rows: {total_stats['total_mapped']}")
    print(f"Unchanged rows: {total_stats['total_unchanged']}")
    print()
    
    if total_stats['all_mappings']:
        print("Mapping Details:")
        for mapping, count in sorted(total_stats['all_mappings'].items()):
            print(f"  {mapping}: {count}")
    
    print()
    print(f"Output files saved alongside original files with '_annotated' suffix.")


if __name__ == '__main__':
    main()
