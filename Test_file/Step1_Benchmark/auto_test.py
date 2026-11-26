import subprocess
import threading
import time
import paramiko
import socket
import sys

sys.stdout.reconfigure(encoding='utf-8')

MODEL_LIST = [
    "qwen3:30b-a3b-thinking-2507-q4_K_M",
    "qwen3:32b-q4_K_M",
    "qwen3:8b-q8_0",
    "qwen3:8b-fp16",
    "deepseek-r1:8b-0528-qwen3-q8_0",
    "deepseek-r1:7b-qwen-distill-fp16",
    "deepseek-r1:8b-llama-distill-fp16",
    "deepseek-r1:8b-0528-qwen3-fp16",
    "deepseek-r1:14b-qwen-distill-q8_0",
    # 在这里加入你所有的模型（think true需要单独写入模型列表）
]

MAIN_TEST_CMD = [
    "python", "-u",  # 无缓冲输出
    "C:\\Users\\chenxi\\Desktop\\Test_file\\Step1_Benchmark\\step1.py",
    "--temperature", "0.1",
    "--max_tokens", "-1",
    "--think", "true",   # 在这里设置think
    "--stream", "false",
    "--serial_port", "COM5"
]

SSH_HOST = "192.168.11.20"
SSH_USER = "chenxi"
SSH_PASS = "54250010"
RESTART_CMD = "sudo systemctl restart ollama"

OLLAMA_HOST = "192.168.11.20"
OLLAMA_PORT = 11434
DELAY_AFTER_RUN = 10
DELAY_AFTER_RESTART = 10
PING_INTERVAL = 10

def run_model(model_name):
    cmd = MAIN_TEST_CMD + ["--model", model_name]
    print(f"\n=== 开始跑分模型: {model_name} ===\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',     # ✅ 防乱码
        errors='replace',     # ✅ 防止 cp932 崩溃
        bufsize=1             # ✅ 实时输出
    )

    def print_output(pipe):
        for line in iter(pipe.readline, ''):
            print(line, end='', flush=True)  # ✅ 实时打印 + 强制刷新
        pipe.close()

    t = threading.Thread(target=print_output, args=(process.stdout,))
    t.start()
    process.wait()
    t.join()

    print(f"\n=== 模型 {model_name} 跑分结束 ===\n")
    return process.returncode

def restart_ollama():
    print(f"=== SSH 连接 {SSH_HOST} 重启 Ollama ===")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(SSH_HOST, username=SSH_USER, password=SSH_PASS)
        stdin, stdout, stderr = ssh.exec_command(RESTART_CMD)
        print(stdout.read().decode())
        err = stderr.read().decode()
        if err:
            print("SSH ERROR:", err)
        ssh.close()
    except Exception as e:
        print("SSH连接失败:", e)

def ping_ollama(host, port, timeout=5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False

if __name__ == "__main__":
    for model in MODEL_LIST:
        run_model(model)
        print(f"等待 {DELAY_AFTER_RUN}s ...")
        time.sleep(DELAY_AFTER_RUN)
        restart_ollama()
        print(f"等待 {DELAY_AFTER_RESTART}s 让 Ollama 启动...")
        time.sleep(DELAY_AFTER_RESTART)
        print("开始检测 Ollama 是否启动...")
        while not ping_ollama(OLLAMA_HOST, OLLAMA_PORT):
            print(f"Ollama 未启动，再等待 {PING_INTERVAL}s ...")
            time.sleep(PING_INTERVAL)
        print("Ollama 已启动，可以进行下一个模型跑分。\n")
    print("\n=== 所有模型跑分完成 ===")