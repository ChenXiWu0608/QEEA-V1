#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Processing Script
Processes all CSV files in Dataset/Step3 directory for column renaming and data mapping.

Mapping Rules (based on workflow_name and question_index):
- customer_bot + 1 -> category=customer_support, subcategory=workflow_cs_simple
- customer_bot + 2 -> category=customer_support, subcategory=workflow_cs_conflict
- customer_bot + 3 -> category=customer_support, subcategory=workflow_cs_prediction
- deep_research + 1-3 -> category=research, subcategory=investigation
- multilingual_translation + 1-2 -> category=translation, subcategory=multilingual
"""

import pandas as pd
import os
import sys
from pathlib import Path
from collections import defaultdict

# Set stdout encoding to utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# Mapping rules definition
# Format: (workflow_name, question_index) -> (new_category, new_subcategory)
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


def scan_csv_files(base_dir: str) -> list[Path]:
    """Recursively scan directory for all CSV files"""
    csv_files = []
    base_path = Path(base_dir)
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)
    
    return sorted(csv_files)


def apply_mapping_to_value(workflow_name, question_index):
    """
    Convert workflow_name and question_index to category and subcategory
    
    Args:
        workflow_name: original workflow_name value (scalar string)
        question_index: original question_index value (scalar int)
    
    Returns:
        (new_category, new_subcategory) tuple
    """
    # Ensure workflow_name is a string
    if workflow_name is None or (isinstance(workflow_name, float) and pd.isna(workflow_name)):
        workflow_name_str = ''
    else:
        workflow_name_str = str(workflow_name).strip()
    
    # Handle question_index - convert to int
    if question_index is None or (isinstance(question_index, float) and pd.isna(question_index)):
        return workflow_name_str, ''
    
    try:
        question_index_int = int(question_index)
    except (ValueError, TypeError):
        try:
            question_index_int = int(str(question_index).strip())
        except (ValueError, TypeError):
            return workflow_name_str, str(question_index).strip() if isinstance(question_index, str) else ''
    
    # Look up mapping rules
    key = (workflow_name_str, question_index_int)
    if key in MAPPING_RULES:
        new_cat, new_sub = MAPPING_RULES[key]
        return new_cat, new_sub
    
    # If no matching rule, keep original workflow_name and question_index
    return workflow_name_str, str(question_index_int)


def process_csv_file(input_path: Path, output_path: Path) -> dict:
    """
    Process a single CSV file
    
    Args:
        input_path: input file path
        output_path: output file path
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'total_rows': 0,
        'mapped_rows': 0,
        'unchanged_rows': 0,
        'mappings': defaultdict(int),
    }
    
    # Read CSV file
    df = pd.read_csv(input_path)
    
    # Check if required columns exist
    if 'workflow_name' not in df.columns:
        raise ValueError(f"Column 'workflow_name' not found in {input_path.name}")
    if 'question_index' not in df.columns:
        raise ValueError(f"Column 'question_index' not found in {input_path.name}")
    
    # Process each row
    new_categories = []
    new_subcategories = []
    original_pairs = []
    
    for idx in range(len(df)):
        workflow_name = df.at[idx, 'workflow_name']
        question_index = df.at[idx, 'question_index']
        
        # Get scalar values
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
        
        # Apply mapping
        new_category, new_subcategory = apply_mapping_to_value(workflow_name_str, question_index_int if question_index_int is not None else question_index_str)
        new_categories.append(new_category)
        new_subcategories.append(new_subcategory)
    
    # Create new columns
    df['category'] = new_categories
    df['subcategory'] = new_subcategories
    
    # Count mappings
    for orig_workflow, orig_index in original_pairs:
        idx = original_pairs.index((orig_workflow, orig_index))
        new_cat = new_categories[idx]
        new_sub = new_subcategories[idx]
        
        # Check if mapping was applied (different from original workflow_name)
        if new_cat != orig_workflow or new_sub != orig_index:
            stats['mapped_rows'] += 1
            stats['mappings'][f"({orig_workflow}, {orig_index}) -> ({new_cat}, {new_sub})"] += 1
        else:
            stats['unchanged_rows'] += 1
    
    stats['total_rows'] = len(df)
    
    # Drop old columns if they exist
    columns_to_drop = []
    if 'workflow_name' in df.columns:
        columns_to_drop.append('workflow_name')
    if 'question_index' in df.columns:
        columns_to_drop.append('question_index')
    # Also drop old category/subcategory if they exist
    # (we're generating new ones from workflow_name and question_index)
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed file
    df.to_csv(output_path, index=False)
    
    return stats


def main():
    """Main function"""
    # Define directory paths
    base_dir = 'Baseline/Step3'
    output_dir = 'Baseline/Step3_processed'
    
    print("=" * 60)
    print("Dataset Processing Script")
    print("=" * 60)
    print()
    
    # Scan CSV files
    print(f"Scanning directory: {base_dir}")
    csv_files = scan_csv_files(base_dir)
    print(f"Found {len(csv_files)} CSV files")
    print()
    
    if not csv_files:
        print("No CSV files found to process. Exiting.")
        return
    
    # Process each file
    total_stats = {
        'total_files': 0,
        'total_rows': 0,
        'total_mapped': 0,
        'total_unchanged': 0,
        'all_mappings': defaultdict(int),
    }
    
    for i, input_path in enumerate(csv_files, 1):
        # Calculate output path
        relative_path = input_path.relative_to(Path(base_dir))
        output_path = Path(output_dir) / relative_path
        
        print(f"[{i}/{len(csv_files)}] Processing: {input_path.name}")
        
        try:
            stats = process_csv_file(input_path, output_path)
            
            total_stats['total_files'] += 1
            total_stats['total_rows'] += stats['total_rows']
            total_stats['total_mapped'] += stats['mapped_rows']
            total_stats['total_unchanged'] += stats['unchanged_rows']
            
            # Merge mapping statistics
            for mapping, count in stats['mappings'].items():
                total_stats['all_mappings'][mapping] += count
            
            print(f"  -> Saved to: {output_path}")
            print(f"  -> Total rows: {stats['total_rows']}, Mapped: {stats['mapped_rows']}, Unchanged: {stats['unchanged_rows']}")
            
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Output summary statistics
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
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
