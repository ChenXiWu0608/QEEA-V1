#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据整理脚本 - Step2
遍历Dataset/Step2/目录下的所有CSV文件，为每条数据追加category、subcategory、language三个字段
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


def build_step2_map(jsonl_data):
    """构建Step2的索引映射"""
    # Step2 JSONL 数据通过 uid 字段唯一标识
    # CSV 中的 test_id (如 test_0000) 对应 JSONL 的索引
    # test_0000 -> index 0, test_0001 -> index 1, ...
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


def get_category_info(test_id, step2_map):
    """根据test_id获取category信息"""
    # 提取test编号
    test_num = int(test_id.split('_')[1])
    
    # Step2: test_0000~test_0023 (24条数据)
    if 0 <= test_num <= 23:
        index = test_num
        if index < len(step2_map):
            item = step2_map[index]
            return {
                'category': clean_text(item.get('category', '')),
                'subcategory': clean_text(item.get('subcategory', '')),
                'language': clean_text(item.get('language', ''))
            }
    
    return {'category': '', 'subcategory': '', 'language': ''}


def process_csv_file(csv_path, output_path, step2_map):
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
        info = get_category_info(test_id, step2_map)
        categories.append(info['category'])
        subcategories.append(info['subcategory'])
        languages.append(info['language'])
    
    # 在status列后插入新列
    status_idx = df.columns.get_loc('status')
    df.insert(status_idx + 1, 'category', categories)  # type: ignore
    df.insert(status_idx + 2, 'subcategory', subcategories)  # type: ignore
    df.insert(status_idx + 3, 'language', languages)  # type: ignore
    
    # 保存到新文件
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Processed: {csv_path} -> {output_path}")


def main():
    """主函数"""
    # 定义数据源路径
    step2_path = 'Test_file/Step2_Long-text-generation/dataset.jsonl'
    
    # 加载JSONL数据
    print("Loading JSONL data...")
    step2_data = load_jsonl_data(step2_path)
    
    # 构建映射
    step2_map = build_step2_map(step2_data)
    
    print(f"Step2: {len(step2_map)} records")
    
    # 定义Dataset/Step2目录
    dataset_step2_path = 'Baseline/Step2'
    
    # 遍历所有CSV文件
    for root, dirs, files in os.walk(dataset_step2_path):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                # 生成输出文件名
                output_file = file.replace('.csv', '_annotated.csv')
                output_path = os.path.join(root, output_file)
                
                # 处理CSV文件
                try:
                    process_csv_file(csv_path, output_path, step2_map)
                except Exception as e:
                    print(f"Failed to process: {csv_path}, Error: {e}")
    
    print("\nAll Step2 files processed!")


if __name__ == '__main__':
    main()