import requests
import json
import time
import csv
import os
import subprocess
from datetime import datetime
import serial
import serial.tools.list_ports
from ctypes import *
from enum import IntEnum
import threading

# ==========================
# 全局配置
# ==========================
BASE_URL = "http://labllm.oldspace.ltd/v1"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)
CSV_PATH = os.path.join(REPORT_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_dify_report.csv")
CSV_HEADERS = [
    "task_id", "start_time_ms", "end_time_ms", "workflow_name",
    "question_index", "prompt", "response", "conversation_id",
    "response_len", "tokens_usage"
]

# ==========================
# 工具函数
# ==========================
# ------------------------ PMD2 定义 ------------------------
PRODUCT_NAME = "PMD2"
PRODUCT_STRING = "ElmorLabs PMD2"
VENDOR_ID = 0xEE
PRODUCT_ID = 0x15
FIRMWARE_VERSION = 0x00
SENSOR_POWER_NUM = 10

class VendorDataStruct(Structure):
    _pack_ = 1
    _fields_ = [('VendorId', c_uint8), ('ProductId', c_uint8), ('FwVersion', c_uint8)]

class PowerSensor(Structure):
    _pack_ = 1
    _fields_ = [('Voltage', c_int16), ('Current', c_int32), ('Power', c_int32)]

class SensorStruct(Structure):
    _pack_ = 1
    _fields_ = [
        ('Vdd', c_uint16),
        ('Tchip', c_int16),
        ('PowerReadings', PowerSensor * SENSOR_POWER_NUM),
        ('EpsPower', c_uint16),
        ('PciePower', c_uint16),
        ('MbPower', c_uint16),
        ('TotalPower', c_uint16),
        ('Ocp', c_uint8 * SENSOR_POWER_NUM)
    ]

class UART_CMD(IntEnum):
    CMD_READ_VENDOR_DATA = 1
    CMD_READ_SENSOR_VALUES = 4
    def toByte(self):
        return self.value.to_bytes(1, byteorder='little')

# ------------------------ PMD2 监控线程 ------------------------
class PMD2Monitor(threading.Thread):
    SENSOR_NAMES = [
        "ATX24", "12V", "5V", "5VSB", "HPWR1",
        "EPS1", "EPS2",
        "PCIE1", "PCIE2", "PCIE3",
    ]

    SENSOR_MAPPING = [0,1,2,3,4,5,6,7,8,9]  # 根据原脚本索引映射
    UNIT_CONVERSION = {
        "V": 1000.0,  # mV -> V
        "I": 1000.0,  # mA -> A
        "P": 1000.0   # mW -> W
    }

    def __init__(self,interval_ms=100):
        super().__init__()
        self.interval = interval_ms / 1000
        self.stop_flag = False
        self.serial_port = "COM5"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"PMD2_DIFY_{timestamp}.csv"
        self.ser = None

    def find_device(self):
        self.ser = serial.Serial(self.serial_port, 115200, timeout=1, rtscts=False, dsrdtr=False)
        self.ser.dtr = True
        buffer = self.ser.read(len(PRODUCT_STRING)+1)
        if buffer != PRODUCT_STRING.encode('ascii') + b'\x00':
            raise RuntimeError("设备欢迎字符串不匹配")
        self.ser.write(UART_CMD.CMD_READ_VENDOR_DATA.toByte())
        buf = self.ser.read(3)
        vendor_data = VendorDataStruct.from_buffer_copy(buf)
        if vendor_data.VendorId != VENDOR_ID or vendor_data.ProductId != PRODUCT_ID:
            raise RuntimeError("设备 Vendor ID 或 Product ID 不匹配")
        print("[PMD2] 设备初始化完成")

    def stop(self):
        self.stop_flag = True

    def run(self):
        self.find_device()
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 构建可读表头
            headers = ["timestamp_ms"]
            # 根据传感器顺序生成表头
            for i, name in zip(range(SENSOR_POWER_NUM), self.SENSOR_NAMES[:SENSOR_POWER_NUM]):
                headers += [f"{name}_V(V)", f"{name}_I(A)", f"{name}_P(W)"]
            writer.writerow(headers)

            while not self.stop_flag:
                t_ms = int(time.time() * 1000)
                try:
                    self.ser.write(UART_CMD.CMD_READ_SENSOR_VALUES.toByte())
                    buf = self.ser.read(sizeof(SensorStruct))
                    if len(buf) != sizeof(SensorStruct):
                        continue
                    sensor = SensorStruct.from_buffer_copy(buf)
                    row = [t_ms]
                    for ps in sensor.PowerReadings:
                        row += [
                            round(ps.Voltage/self.UNIT_CONVERSION["V"], 3),
                            round(ps.Current/self.UNIT_CONVERSION["I"], 3),
                            round(ps.Power/self.UNIT_CONVERSION["P"], 3)
                        ]
                    writer.writerow(row)
                    f.flush()
                except Exception as e:
                    print(f"[PMD2] 采集异常: {e}")
                time.sleep(self.interval)

# ==========================
# 各任务对应的 API 端点
# ==========================
TASK_ENDPOINTS = {
    1: f"{BASE_URL}/chat-messages",
    2: f"{BASE_URL}/chat-messages",
    3: f"{BASE_URL}/workflows/run",
    4: f"{BASE_URL}/chat-messages",
}

# ==========================
# Dify 消息发送函数
# ==========================
def send_dify_message(api_url, api_key, user_input, conversation_id=None):
    """
    向 Dify API 发送消息并返回 (answer, conversation_id, usage)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": {},
        "query": user_input,
        "response_mode": "blocking",
        "conversation_id": conversation_id,
        "user": "test-user"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Dify 返回格式
        answer = data.get("answer", "")
        conv_id = data.get("conversation_id", None)
        usage = data.get("metadata", {}).get("usage", {})
        return answer, conv_id, usage

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return "(请求失败)", conversation_id, {}

# ==========================
# 各任务逻辑
# ==========================
def write_report(rows):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(CSV_HEADERS)
        writer.writerows(rows)

def wait_for_env(seconds=10, note="等待环境初始化"):
    print(f"\n⏸️ {note}中 ({seconds}s)...")
    for i in range(seconds, 0, -1):
        print(f"  ⏳ {i}s 剩余", end="\r")
        time.sleep(1)
    print("✅ 继续执行\n")

def run_task1(task, buffer_sec=10):
    print(f"▶️ 执行任务1: {task['workflow_name']}")
    conv_id = None
    results = []
    api_url = TASK_ENDPOINTS[1]

    for idx, question in enumerate(task["questions"], start=1):
        start_ms = int(time.time() * 1000)
        answer, conv_id, _ = send_dify_message(
            api_url, task["api_key"], question, conversation_id=conv_id
        )
        end_ms = int(time.time() * 1000)

        results.append([
            task["task_id"],
            start_ms,
            end_ms,
            task["workflow_name"],
            idx,
            question,
            answer,
            conv_id,
            len(answer),
            ""  # tokens_usage
        ])
        print(f"🗨️客服对话Q{idx}完成 | 输出长度: {len(answer)}\n⏱ {start_ms} → {end_ms}ms\n{question[:100]}...\n💬 A: {answer[:120]}...")

        if idx < len(task["questions"]):
            wait_for_env(buffer_sec, note=f"子任务{idx}完成，等待 {buffer_sec}s")

    write_report(results)


def run_task2(task):
    print(f"▶️ 执行任务2: {task['workflow_name']}")
    results = []
    conv_id = None
    api_url = TASK_ENDPOINTS[2]

    for idx, question in enumerate(task["questions"], start=1):
        start_ms = int(time.time() * 1000)
        answer, conv_id, _ = send_dify_message(api_url, task["api_key"], question)
        end_ms = int(time.time() * 1000)

        results.append([
            task["task_id"],
            start_ms,
            end_ms,
            task["workflow_name"],
            idx,
            question,
            answer,
            conv_id,
            len(answer),
            ""
        ])
        print(f"🧠deep_research Q{idx}完成 | 输出长度: {len(answer)}\n⏱ {start_ms} → {end_ms}ms\n{question[:100]}...\n💬 A: {answer[:120]}...\n⏱ {start_ms} → {end_ms}ms\n")

    write_report(results)


def run_task3(task):
    print(f"▶️ 执行任务3: {task['workflow_name']}")
    results = []
    conv_id = None
    api_url = TASK_ENDPOINTS[3]

    for sub_idx, q in enumerate(task["questions"], start=1):
        source_lang = q["source_lang"]
        target_lang = q["target_lang"]
        source_text = q["source_text"]

        payload = {
            "inputs": {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_text": source_text
            },
            "response_mode": "blocking",
            "user": "abc-123"
        }

        headers = {
            "Authorization": f"Bearer {task['api_key']}",
            "Content-Type": "application/json"
        }
        
        start_ms = int(time.time() * 1000)
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        data = response.json()
        answer = data.get("outputs", "")
        conv_id = data.get("task_id", None)
        end_ms = int(time.time() * 1000)

        results.append([
            task["task_id"],
            start_ms,
            end_ms,
            task["workflow_name"],
            sub_idx,
            f"{source_lang}→{target_lang}: {source_text[:100]}...",
            answer,
            conv_id,
            len(answer),
            ""
        ])
        print(f"💬 翻译完成 | 输出长度: {len(answer)} | ⏱ {start_ms} → {end_ms}ms\n")

        if sub_idx < len(task["questions"]):
            wait_for_env(10, note=f"子任务{sub_idx}完成，准备下一个翻译任务")

    write_report(results)


def run_task4(task):
    print(f"▶️ 执行任务4: {task['workflow_name']}")
    results = []
    api_url = TASK_ENDPOINTS[4]

    for chain_idx, prompt_chain in enumerate(task["questions"], start=1):
        print(f"\n📂 第 {chain_idx} 组任务（{len(prompt_chain)} 步）")
        conv_id = None

        for step_idx, prompt in enumerate(prompt_chain, start=1):
            start_ms = int(time.time() * 1000)
            answer, conv_id, _ = send_dify_message(
                api_url, task["api_key"], prompt, conversation_id=conv_id
            )
            end_ms = int(time.time() * 1000)

            results.append([
                task["task_id"],
                start_ms,
                end_ms,
                task["workflow_name"],
                f"{chain_idx}-{step_idx}",
                prompt,
                answer,
                conv_id,
                len(answer),
                ""
            ])
            print(f"💬编码完成 | 输出长度: {len(answer)} | ⏱ {start_ms} → {end_ms}ms\n")

        if chain_idx < len(task["questions"]):
            wait_for_env(10, note=f"第{chain_idx}组任务完成，准备下一组")

    write_report(results)
    print(f"✅ 任务4完成，共 {len(results)} 条结果。\n")

# ==========================
# 主程序入口
# ==========================
def main():
    print("🚀 开始执行所有任务")

    # PMD2 监控线程
    pmd2_monitor = PMD2Monitor(interval_ms=100)
    pmd2_monitor.start()
    print("[INFO] PMD2 监控已启动，等待3秒再开始测试...")
    time.sleep(3)

    # 执行 Dify 任务（保持原任务不变）
    with open("tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)
    
    # 处理任务
    for idx, task in enumerate(tasks, start=1):
        if task["task_id"] == 1:
            run_task1(task)
        elif task["task_id"] == 2:
            run_task2(task)
        elif task["task_id"] == 3:
            run_task3(task)
        elif task["task_id"] == 4:
            run_task4(task)

    # 关闭串口连接
    pmd2_monitor.stop()
    pmd2_monitor.join()
    if pmd2_monitor.ser:
        pmd2_monitor.ser.close()

    print("所有任务执行完毕.")

if __name__ == "__main__":
    main()
