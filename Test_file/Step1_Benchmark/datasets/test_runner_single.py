# -*- coding: utf-8 -*-
"""
流式测试运行器 - 每条提示词独立进程执行，超时跳过
"""
DEBUG = False
import requests
import json
import time
import os
import csv
from datetime import datetime
from multiprocessing import Process, Queue

class TestRunner:
    def __init__(self, model, temperature, max_tokens, think, stream):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think
        self.stream = stream
        self.timeout = 180  # 每条提示词最大执行时间

        self.ollama_url = "http://localhost:11434/api/generate"
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_csv = os.path.join(self.report_dir, f"test_report{model}_{timestamp}.csv")

        # CSV 表头
        with open(self.report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "test_id", "timestamp", "status", "prompt", "response", "thinking",
                "execution_time_s", "api_error", "total_duration",
                "load_duration", "prompt_eval_duration", "eval_duration"
            ])

    def _worker(self, prompt, queue):
        """子进程调用 Ollama API"""
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": self.stream,
                "think": self.think,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.max_tokens
                }
            }
            if DEBUG:
                print(f"[DEBUG] 请求内容: {json.dumps(data, ensure_ascii=False)}")

            session = requests.Session()
            response = session.post(self.ollama_url, json=data, timeout=300)
            try:
                response.raise_for_status()
                result = response.json()
            finally:
                response.close()

            queue.put({
                "status": "success",
                "response": result.get("response", ""),
                "thinking": result.get("thinking", ""),
                "total_duration": result.get("total_duration", 0),
                "load_duration": result.get("load_duration", 0),
                "prompt_eval_duration": result.get("prompt_eval_duration", 0),
                "eval_duration": result.get("eval_duration", 0),
                "error": ""
            })
        except Exception as e:
            queue.put({
                "status": "error",
                "response": "",
                "thinking": "",
                "error": str(e),
                "total_duration": "",
                "load_duration": "",
                "prompt_eval_duration": "",
                "eval_duration": ""
            })

    def run_single_test(self, prompt, test_id):
        print(f"\n[{test_id}] 开始推理：{prompt[:60]}...")

        start_time = time.time()
        queue = Queue()
        p = Process(target=self._worker, args=(prompt, queue))
        p.start()
        p.join(timeout=self.timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            result = {
                "status": "timeout",
                "response": "",
                "thinking": "",
                "error": f"超时 {self.timeout}s",
                "total_duration": "",
                "load_duration": "",
                "prompt_eval_duration": "",
                "eval_duration": ""
            }
        else:
            result = queue.get()

        end_time = time.time()
        exec_time = round(end_time - start_time, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 替换换行符
        prompt_sanitized = prompt.replace("\n", " ").replace("\r", " ")
        response_sanitized = result.get("response", "").replace("\n", " ").replace("\r", " ")
        thinking_sanitized = result.get("thinking", "").replace("\n", " ").replace("\r", " ")

        row = [
            test_id,
            timestamp,
            result.get("status", "unknown"),
            prompt_sanitized,
            response_sanitized,
            thinking_sanitized,
            exec_time,
            result.get("error", ""),
            result.get("total_duration", ""),
            result.get("load_duration", ""),
            result.get("prompt_eval_duration", ""),
            result.get("eval_duration", "")
        ]

        # 写入 CSV
        with open(self.report_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[{test_id}] 完成 ✅ 耗时 {exec_time}s 状态: {result.get('status')}")

    def run_tests(self, prompts):
        print(f"共 {len(prompts)} 条提示词，将逐一推理并写入报告：{self.report_csv}")
        for i, prompt in enumerate(prompts):
            test_id = f"test_{i:04d}"
            self.run_single_test(prompt, test_id)
        print("\n全部测试完成 ✅")
