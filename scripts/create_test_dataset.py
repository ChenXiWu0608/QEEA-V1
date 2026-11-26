"""
从Dataset文件夹中随机采样创建测试数据集
- Step1: 随机选取50个task
- Step2: 随机选取25个task
- Step3: 随机选取25个task（排除workflow_name为code_generation的行）
总计: 100个task

输出CSV的表头涵盖Step1-3的所有列：
- source_step: 标识数据来源（Step1/Step2/Step3）
- Step1/Step2特有列: test_id, status, thinking, api_error, total_duration_ms, load_duration_ms, prompt_eval_duration_ms, eval_duration_ms, prompt_eval_count, eval_count
- Step3特有列: task_id, workflow_name, question_index, conversation_id, response_len, tokens_usage, Availability
- 共有列: start_time_ms, end_time_ms, category, subcategory, language, prompt, response
"""

import os
import csv
import random


def get_all_csv_files(dataset_dir, step_name):
    """获取指定Step目录下所有的csv文件"""
    step_dir = os.path.join(dataset_dir, step_name)
    csv_files = []
    for root, dirs, files in os.walk(step_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def read_csv_rows(csv_file_path):
    """读取CSV文件中的所有数据行（跳过表头）"""
    rows = []
    headers = None
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 读取表头
        for row in reader:
            if row:  # 跳过空行
                rows.append(row)
    return headers, rows


def sample_from_step(dataset_dir, step_name, sample_count, exclude_code_generation=False):
    """从指定Step中随机采样指定数量的task
    
    Args:
        dataset_dir: 数据集目录
        step_name: Step名称 (Step1/Step2/Step3)
        sample_count: 采样数量
        exclude_code_generation: 是否排除workflow_name为code_generation的行（仅适用于Step3）
    """
    csv_files = get_all_csv_files(dataset_dir, step_name)
    
    if not csv_files:
        print(f"警告: {step_name} 中没有找到CSV文件")
        return None, []
    
    print(f"{step_name}: 找到 {len(csv_files)} 个CSV文件")
    
    all_rows = []
    headers = None
    
    for csv_file in csv_files:
        file_headers, file_rows = read_csv_rows(csv_file)
        if headers is None:
            headers = file_headers
        all_rows.extend(file_rows)
    
    print(f"{step_name}: 总共读取 {len(all_rows)} 行数据")
    
    # 如果是Step3且需要排除code_generation，则过滤
    if exclude_code_generation:
        filtered_rows = []
        for row in all_rows:
            # Step3的workflow_name在第4列（索引3）
            workflow_name = row[3] if len(row) > 3 else ''
            if workflow_name != 'code_generation':
                filtered_rows.append(row)
        print(f"{step_name}: 排除code_generation后剩余 {len(filtered_rows)} 行数据")
        all_rows = filtered_rows
    
    if len(all_rows) < sample_count:
        print(f"警告: {step_name} 中的数据行数({len(all_rows)})少于采样数({sample_count})，将取全部数据")
        sampled_rows = all_rows
    else:
        sampled_rows = random.sample(all_rows, sample_count)
    
    print(f"{step_name}: 已随机选取 {len(sampled_rows)} 行")
    
    return headers, sampled_rows


def get_safe_value(row, index, default=''):
    """安全获取行中的值，索引超出范围时返回默认值"""
    if 0 < index <= len(row):
        return row[index - 1]
    return default


def create_test_dataset(output_file='Test_dataset.csv', seed=42):
    """创建100问的测试数据集"""
    random.seed(seed)
    
    # 获取脚本所在目录的父目录作为项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_dir, 'Dataset')
    
    print(f"数据集目录: {dataset_dir}")
    print("=" * 50)
    
    # 从各Step采样（Step3排除code_generation）
    step1_headers, step1_rows = sample_from_step(dataset_dir, 'Step1', 50)
    step2_headers, step2_rows = sample_from_step(dataset_dir, 'Step2', 25)
    step3_headers, step3_rows = sample_from_step(dataset_dir, 'Step3', 25, exclude_code_generation=True)
    
    # 统一的输出表头，涵盖Step1-3的所有列
    output_headers = [
        # 来源标识
        'source_step',
        # 共有列
        'test_id',
        'start_time_ms',
        'end_time_ms',
        'category',
        'subcategory',
        'language',
        'prompt',
        'response',
        # Step1/Step2特有列
        'status',
        'thinking',
        'api_error',
        'total_duration_ms',
        'load_duration_ms',
        'prompt_eval_duration_ms',
        'eval_duration_ms',
        'prompt_eval_count',
        'eval_count',
        # Step3特有列
        'workflow_name',
        'question_index',
        'conversation_id',
        'response_len',
        'tokens_usage',
        'Availability'
    ]
    
    # Step1/Step2的列索引映射
    # test_id(1), start_time_ms(2), end_time_ms(3), status(4), category(5), 
    # subcategory(6), language(7), prompt(8), response(9), thinking(10),
    # api_error(11), total_duration_ms(12), load_duration_ms(13),
    # prompt_eval_duration_ms(14), eval_duration_ms(15), prompt_eval_count(16), eval_count(17)
    
    # Step3的列索引映射
    # task_id(1), start_time_ms(2), end_time_ms(3), workflow_name(4), question_index(5),
    # prompt(6), response(7), conversation_id(8), response_len(9), tokens_usage(10),
    # Availability(11), category(12), subcategory(13), language(14)
    
    processed_rows = []
    
    # Step1数据
    for row in step1_rows:
        processed_row = [
            'Step1',                              # source_step
            get_safe_value(row, 1),               # test_id
            get_safe_value(row, 2),               # start_time_ms
            get_safe_value(row, 3),               # end_time_ms
            get_safe_value(row, 5),               # category
            get_safe_value(row, 6),               # subcategory
            get_safe_value(row, 7),               # language
            get_safe_value(row, 8),               # prompt
            get_safe_value(row, 9),               # response
            get_safe_value(row, 4),               # status
            get_safe_value(row, 10),              # thinking
            get_safe_value(row, 11),              # api_error
            get_safe_value(row, 12),              # total_duration_ms
            get_safe_value(row, 13),              # load_duration_ms
            get_safe_value(row, 14),              # prompt_eval_duration_ms
            get_safe_value(row, 15),              # eval_duration_ms
            get_safe_value(row, 16),              # prompt_eval_count
            get_safe_value(row, 17),              # eval_count
            '',                                   # workflow_name (Step3 only)
            '',                                   # question_index (Step3 only)
            '',                                   # conversation_id (Step3 only)
            '',                                   # response_len (Step3 only)
            '',                                   # tokens_usage (Step3 only)
            ''                                    # Availability (Step3 only)
        ]
        processed_rows.append(processed_row)
    
    # Step2数据（格式与Step1相同）
    for row in step2_rows:
        processed_row = [
            'Step2',                              # source_step
            get_safe_value(row, 1),               # test_id
            get_safe_value(row, 2),               # start_time_ms
            get_safe_value(row, 3),               # end_time_ms
            get_safe_value(row, 5),               # category
            get_safe_value(row, 6),               # subcategory
            get_safe_value(row, 7),               # language
            get_safe_value(row, 8),               # prompt
            get_safe_value(row, 9),               # response
            get_safe_value(row, 4),               # status
            get_safe_value(row, 10),              # thinking
            get_safe_value(row, 11),              # api_error
            get_safe_value(row, 12),              # total_duration_ms
            get_safe_value(row, 13),              # load_duration_ms
            get_safe_value(row, 14),              # prompt_eval_duration_ms
            get_safe_value(row, 15),              # eval_duration_ms
            get_safe_value(row, 16),              # prompt_eval_count
            get_safe_value(row, 17),              # eval_count
            '',                                   # workflow_name (Step3 only)
            '',                                   # question_index (Step3 only)
            '',                                   # conversation_id (Step3 only)
            '',                                   # response_len (Step3 only)
            '',                                   # tokens_usage (Step3 only)
            ''                                    # Availability (Step3 only)
        ]
        processed_rows.append(processed_row)
    
    # Step3数据（格式不同）
    for row in step3_rows:
        processed_row = [
            'Step3',                              # source_step
            get_safe_value(row, 1),               # test_id (task_id)
            get_safe_value(row, 2),               # start_time_ms
            get_safe_value(row, 3),               # end_time_ms
            get_safe_value(row, 12),              # category
            get_safe_value(row, 13),              # subcategory
            get_safe_value(row, 14),              # language
            get_safe_value(row, 6),               # prompt
            get_safe_value(row, 7),               # response
            '',                                   # status (Step1/2 only)
            '',                                   # thinking (Step1/2 only)
            '',                                   # api_error (Step1/2 only)
            '',                                   # total_duration_ms (Step1/2 only)
            '',                                   # load_duration_ms (Step1/2 only)
            '',                                   # prompt_eval_duration_ms (Step1/2 only)
            '',                                   # eval_duration_ms (Step1/2 only)
            '',                                   # prompt_eval_count (Step1/2 only)
            '',                                   # eval_count (Step1/2 only)
            get_safe_value(row, 4),               # workflow_name
            get_safe_value(row, 5),               # question_index
            get_safe_value(row, 8),               # conversation_id
            get_safe_value(row, 9),               # response_len
            get_safe_value(row, 10),              # tokens_usage
            get_safe_value(row, 11)               # Availability
        ]
        processed_rows.append(processed_row)
    
    # 写入输出CSV
    output_path = os.path.join(project_dir, output_file)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_headers)
        writer.writerows(processed_rows)
    
    print("=" * 50)
    print(f"测试数据集已创建: {output_path}")
    print(f"总计: {len(processed_rows)} 个task")
    print(f"  - Step1: {len(step1_rows)} 个")
    print(f"  - Step2: {len(step2_rows)} 个")
    print(f"  - Step3: {len(step3_rows)} 个")
    print(f"\n输出表头 ({len(output_headers)} 列):")
    print(', '.join(output_headers))


if __name__ == '__main__':
    # 设置随机种子以确保可复现性
    # 可以使用不同的种子来生成不同的测试集
    create_test_dataset(seed=42)