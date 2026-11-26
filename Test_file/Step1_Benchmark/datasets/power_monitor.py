#!/usr/bin/env python3
import psutil
import csv
import time
import subprocess
import os
import traceback
import signal
import argparse

INTERVAL = 1
running = True

def parse_args():
    parser = argparse.ArgumentParser(description="系统性能监控")
    parser.add_argument("--model", type=str, default="unknown_model", help="当前运行的模型名称")
    return parser.parse_args()

def sanitize_filename(name):
    # 去掉文件名中不合法的字符（如 : / \ 等）
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)

def setup_logfile(model_name):
    REPORT_DIR = os.path.join(os.getcwd(), "reports")
    os.makedirs(REPORT_DIR, exist_ok=True)
    safe_model = sanitize_filename(model_name)
    logfile = os.path.join(REPORT_DIR, f"system_{safe_model}_log.csv")
    return logfile

def signal_handler(sig, frame):
    global running
    print(f"\n[INFO] 收到信号 {sig}, 准备退出...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def write_header_if_needed(logfile):
    if not os.path.isfile(logfile):
        with open(logfile, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "cpu_percent",
                "cpu_temp_C",
                "mem_used_MB",
                "net_sent_MB",
                "net_recv_MB",
                "gpu_index",
                "gpu_util_percent",
                "gpu_mem_used_MB",
                "gpu_power_W",
                "gpu_fan_percent",
                "gpu_temp_C"
            ])

net_prev = psutil.net_io_counters()

def get_network_io():
    global net_prev
    try:
        net_current = psutil.net_io_counters()
        sent = (net_current.bytes_sent - net_prev.bytes_sent) / (1024**2)
        recv = (net_current.bytes_recv - net_prev.bytes_recv) / (1024**2)
        net_prev = net_current
        return sent, recv
    except:
        return 0, 0

def get_cpu_temp():
    try:
        output = subprocess.check_output(["sensors"], text=True)
        for line in output.splitlines():
            if line.startswith("Tctl:"):
                return float(line.split()[1].replace("+","").replace("°C",""))
    except:
        return "N/A"

def get_gpu_info():
    gpu_data = []
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,power.draw,fan.speed,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True)
        for line in result.stdout.strip().split("\n"):
            parts = line.split(", ")
            while len(parts) < 6:
                parts.append("N/A")
            gpu_data.append(parts)
    except:
        gpu_data.append(["0","N/A","N/A","N/A","N/A","N/A"])
    return gpu_data

def write_log(logfile):
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_temp = get_cpu_temp()
        mem_used = round(psutil.virtual_memory().used / (1024**2), 2)
        net_sent, net_recv = get_network_io()
        gpu_infos = get_gpu_info()

        with open(logfile, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for gpu in gpu_infos:
                writer.writerow([
                    timestamp,
                    cpu_percent,
                    cpu_temp,
                    mem_used,
                    net_sent,
                    net_recv,
                    *gpu
                ])
            f.flush()
    except Exception as e:
        print("[ERROR] 写入日志失败:", e)
        traceback.print_exc()

def main():
    args = parse_args()
    logfile = setup_logfile(args.model)
    print(f"[INFO] 电源监控启动，每 {INTERVAL}s 写入 {logfile}")
    write_header_if_needed(logfile)
    try:
        while running:
            start = time.time()
            write_log(logfile)
            elapsed = time.time() - start
            time.sleep(max(0, INTERVAL - elapsed))
    finally:
        print("[INFO] 电源监控退出前写入最后一条")
        write_log(logfile)

if __name__ == "__main__":
    main()
