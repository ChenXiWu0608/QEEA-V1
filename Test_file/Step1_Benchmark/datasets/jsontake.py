import csv
import json

csv_file = "gpqa_main.csv"
jsonl_file = "arena_v5_final.jsonl"

with open(csv_file, encoding="utf-8") as f_in, open(jsonl_file, "w", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)
    count = 0
    for idx, row in enumerate(reader, start=1):
        sample = {
            "uid": row.get("Record ID", str(idx)),
            "Question": row.get("Question", ""),
            "Correct Answer": row.get("Correct Answer", ""),
            "Incorrect Answer1": row.get("Incorrect Answer 1", ""),
            "Incorrect Answer2": row.get("Incorrect Answer 2", ""),
            "Incorrect Answer3": row.get("Incorrect Answer 3", "")
        }
        f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
        count += 1
        if count <= 3:  # 打印前 3 条，确认读取正常
            print(sample)

print(f"转换完成，共处理 {count} 条题目，生成文件：{jsonl_file}")
