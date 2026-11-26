#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据整理脚本
遍历Dataset/Step1/目录下的所有CSV文件，为每条数据追加category、subcategory、language三个字段
"""

import os
import json
import pandas as pd
from pathlib import Path


def load_jsonl_data(jsonl_path):
    """加载JSONL文件，返回列表"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def build_arena_map(jsonl_data):
    """构建arena_hard_v2的索引到category信息的映射"""
    return jsonl_data


def build_livecode_map(jsonl_data):
    """构建LiveCodeBench的索引到category信息的映射"""
    return jsonl_data


def build_gpqa_map(jsonl_data):
    """构建gpqa的索引到category信息的映射"""
    return jsonl_data


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


def get_category_info(test_id, arena_map, livecode_map, gpqa_map):
    """根据test_id获取category信息"""
    # 提取test编号
    test_num = int(test_id.split('_')[1])
    
    # arena_hard_v2: test0000~test0099
    if 0 <= test_num <= 99:
        index = test_num
        if index < len(arena_map):
            item = arena_map[index]
            return {
                'category': clean_text(item.get('category', '')),
                'subcategory': clean_text(item.get('subcategory', '')),
                'language': clean_text(item.get('language', ''))
            }
    
    # LiveCodeBench: test0100~test0149
    elif 100 <= test_num <= 149:
        index = test_num - 100
        if index < len(livecode_map):
            item = livecode_map[index]
            return {
                'category': 'LiveCodeBench',
                'subcategory': clean_text(item.get('difficulty', '')),
                'language': 'english'
            }
    
    # GPQA: test0150~test0199
    elif 150 <= test_num <= 199:
        index = test_num - 150
        if index < len(gpqa_map):
            item = gpqa_map[index]
            return {
                'category': 'GPQA',
                'subcategory': clean_text(item.get('Correct Answer', '')),
                'language': 'english'
            }
    
    return {'category': '', 'subcategory': '', 'language': ''}


def process_csv_file(csv_path, output_path, arena_map, livecode_map, gpqa_map):
    """处理单个CSV文件，添加category信息"""
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查是否有test_id列
    if 'test_id' not in df.columns:
        print(f"Skipping (no test_id column): {csv_path}")
        return
    
    # 添加新列
    categories = []
    subcategories = []
    languages = []
    
    for _, row in df.iterrows():
        test_id = row['test_id']
        info = get_category_info(test_id, arena_map, livecode_map, gpqa_map)
        categories.append(info['category'])
        subcategories.append(info['subcategory'])
        languages.append(info['language'])
    
    # 在status列后插入新列
    status_idx = df.columns.get_loc('status')
    df.insert(status_idx + 1, 'category', categories)  # type: ignore
    df.insert(status_idx + 2, 'subcategory', subcategories)  # type: ignore
    df.insert(status_idx + 3, 'language', languages)  # type: ignore
    
    # ���存到新文件
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Processed: {csv_path} -> {output_path}")


def main():
    """主函数"""
    # 定义数据源路径
    arena_hard_path = 'Test_file/Step1_Benchmark/datasets/arena_hard_v2_picked.jsonl'
    livecode_path = 'Test_file/Step1_Benchmark/datasets/LiveCodeBench_v1_picked.jsonl'
    gpqa_path = 'Test_file/Step1_Benchmark/datasets/gpqa_picked.jsonl'
    
    # 加载JSONL数据
    print("Loading JSONL data...")
    arena_data = load_jsonl_data(arena_hard_path)
    livecode_data = load_jsonl_data(livecode_path)
    gpqa_data = load_jsonl_data(gpqa_path)
    
    # 构建映射
    arena_map = build_arena_map(arena_data)
    livecode_map = build_livecode_map(livecode_data)
    gpqa_map = build_gpqa_map(gpqa_data)
    
    print(f"arena_hard_v2: {len(arena_map)} records")
    print(f"LiveCodeBench: {len(livecode_map)} records")
    print(f"gpqa: {len(gpqa_map)} records")
    
    # 定义Dataset/Step1目录
    dataset_step1_path = 'Baseline/Step1'
    
    # 遍历所有CSV文件
    for root, dirs, files in os.walk(dataset_step1_path):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                # 生成输出文件名
                output_file = file.replace('.csv', '_annotated.csv')
                output_path = os.path.join(root, output_file)
                
                # 处理CSV文件
                try:
                    process_csv_file(csv_path, output_path, arena_map, livecode_map, gpqa_map)
                except Exception as e:
                    print(f"Failed to process: {csv_path}, Error: {e}")
    
    print("\nAll files processed!")


if __name__ == '__main__':
    main()
