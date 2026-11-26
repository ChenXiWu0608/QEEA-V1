# -*- coding: utf-8 -*-
"""
LLM 测试系统 + PMD2 硬件监控集成
"""

import argparse, sys, os, json, random, time, threading, csv, signal, traceback
from multiprocessing import Process, Queue
import requests
import serial
from ctypes import *
from enum import IntEnum
import numpy as np

# ------------------------ 全局控制 ------------------------
stop_flag = False  # True 表示用户希望退出测试

# ------------------------ 信号处理 ------------------------
def signal_handler(sig, frame):
    global stop_flag
    print(f"\n[INFO] 收到信号 {sig}, 准备退出...")
    stop_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ------------------------ 命令行参数解析 ------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='LLM能耗测试系统')
    parser.add_argument('--model', type=str, required=True, help='要测试的模型名称')
    parser.add_argument('--think', type=str, default='false', choices=['true','false'], help='是否开启链式思维')
    parser.add_argument('--stream', type=str, default='false', choices=['true','false'], help='是否开启流式输出')
    parser.add_argument('--max_tokens', type=int, default=-1, help='最大token数')
    parser.add_argument('--temperature', type=float, default=0.5, help='温度参数')
    parser.add_argument('--serial_port', type=str, required=True, help='PMD2 串口号，例如 COM3')
    return parser.parse_args()

# ------------------------ 数据集加载 ------------------------
def load_jsonl_dataset(file_path, key="prompt"):
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    if key in data:
                        prompts.append(data[key])
    except Exception as e:
        print(f"加载 {file_path} 出错: {e}")
    return prompts

def load_all_datasets():
    # 当前 Python 文件同目录下的 dataset.jsonl
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    all_prompts = []

    if os.path.exists(dataset_path):
        prompts = load_jsonl_dataset(dataset_path)
        all_prompts.extend(prompts)
        print(f"[INFO] 加载 dataset.jsonl: {len(prompts)} 条提示词")
    else:
        print(f"[WARN] dataset.jsonl 文件不存在")

    print(f"[INFO] 数据集总计 {len(all_prompts)} 条提示词")
    return all_prompts

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

    def __init__(self, model_name, think, serial_port, interval_ms=100):
        super().__init__()
        self.interval = interval_ms / 1000
        self.stop_flag = False
        self.serial_port = serial_port
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_model = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in model_name)
        self.csv_path = f"PMD2_{timestamp}_{safe_model}_{think}.csv"
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

# ------------------------ 测试运行器 ------------------------
class TestRunner:
    def __init__(self, model, temperature, max_tokens, think, stream):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think
        self.stream = stream
        self.timeout = 240
        self.ollama_url = "http://192.168.11.20:11434/api/generate"
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)
        safe_model = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in model)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.report_csv = os.path.join(self.report_dir, f"test_{safe_model}_{think}_{timestamp}.csv")
        with open(self.report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "test_id","start_time_ms","end_time_ms","status","prompt","response","thinking","api_error",
                "total_duration_ms","load_duration_ms","prompt_eval_duration_ms","eval_duration_ms",
                "prompt_eval_count","eval_count"
            ])

    def _worker(self, prompt, queue):
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": self.stream,
                "think": self.think,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
            }
            session = requests.Session()
            response = session.post(self.ollama_url, json=data, timeout=self.timeout)
            try:
                response.raise_for_status()
            finally:
                response.close()
            result = response.json()
            queue.put({
                "status": "success",
                "response": result.get("response", ""),
                "thinking": result.get("thinking", ""),
                "total_duration": result.get("total_duration", 0) / 1e6,
                "load_duration": result.get("load_duration", 0) / 1e6,
                "prompt_eval_duration": result.get("prompt_eval_duration", 0) / 1e6,
                "eval_duration": result.get("eval_duration", 0) / 1e6,
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "eval_count": result.get("eval_count", 0),
                "error": ""
            })
        except Exception as e:
            queue.put({
                "status": "error",
                "response": "",
                "thinking": "",
                "error": str(e),
                "total_duration": "", "load_duration": "",
                "prompt_eval_duration": "", "eval_duration": "",
                "prompt_eval_count": "", "eval_count": ""
            })

    def run_single_test(self, prompt, test_id):
        global stop_flag
        print(f"\n[{test_id}] 开始推理：{prompt[:60]}...")
        start_time_ms = int(time.time() * 1000)
        queue = Queue()
        p = Process(target=self._worker, args=(prompt, queue))
        p.start()
        result = None
        while p.is_alive():
            p.join(timeout=1)
            if stop_flag:
                try:
                    p.terminate()
                    p.join()
                except:
                    pass
                result = {"status":"stopped","response":"","thinking":"","error":"用户中断",
                          "total_duration":"","load_duration":"","prompt_eval_duration":"",
                          "eval_duration":"","prompt_eval_count":"","eval_count":""}
                break
        if result is None:
            try:
                result = queue.get(timeout=2)
            except Exception as e:
                result = {"status":"error","response":"","thinking":"",
                          "error":f"未获取到子进程结果或队列超时: {str(e)}",
                          "total_duration":"","load_duration":"","prompt_eval_duration":"",
                          "eval_duration":"","prompt_eval_count":"","eval_count":""}
        end_time_ms = int(time.time() * 1000)
        row = [
            test_id, start_time_ms, end_time_ms, result.get("status", "unknown"),
            prompt.replace("\n"," ").replace("\r"," "),
            result.get("response", "").replace("\n"," ").replace("\r"," "),
            result.get("thinking", "").replace("\n"," ").replace("\r"," "),
            result.get("error", ""),
            result.get("total_duration", ""), result.get("load_duration", ""),
            result.get("prompt_eval_duration", ""), result.get("eval_duration", ""),
            result.get("prompt_eval_count", ""), result.get("eval_count", "")
        ]
        with open(self.report_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        print(f"[{test_id}] 完成 ✅ 开始:{start_time_ms}ms 结束:{end_time_ms}ms 状态: {result.get('status')}")
        return result

    def run_tests(self, prompts):
        global stop_flag
        print(f"共 {len(prompts)} 条提示词\n将逐一推理并写入报告：{self.report_csv}")
        results = []
        for i, prompt in enumerate(prompts):
            if stop_flag:
                print("[INFO] stop_flag 激活，提前退出测试循环")
                break
            test_id = f"test_{i:04d}"
            result = self.run_single_test(prompt, test_id)
            results.append(result)
        print("\n全部测试完成 ✅")
        return results

# ------------------------ 主函数 ------------------------
def main():
    global stop_flag
    args = parse_arguments()
    print("=== LLM测试系统启动 ===")
    print(f"模型: {args.model}")

    # 自动加载三个数据集
    prompts = load_all_datasets()
    if not prompts:
        print("错误: 数据集为空或无法读取")
        return

    # PMD2 监控线程
    pmd2_monitor = PMD2Monitor(model_name=args.model, think=args.think, serial_port=args.serial_port, interval_ms=100)
    pmd2_monitor.start()
    print("[INFO] PMD2 监控已启动，等待3秒再开始测试...")
    time.sleep(3)

    think_bool = (args.think.lower() == "true")
    stream_bool = (args.stream.lower() == "true")
    runner = TestRunner(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        think=think_bool,
        stream=stream_bool
    )

    try:
        results = runner.run_tests(prompts)
        print(f"\n=== 测试完成 === 成功测试: {len([r for r in results if r['status']=='success'])} / 总数 {len(prompts)}")
    except KeyboardInterrupt:
        print("\n用户中断测试")
        stop_flag = True
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        # ✅ 修改部分：PMD2 延迟停止逻辑
        print("[INFO] 测试已结束，PMD2 将继续记录 5 秒 ...")
        time.sleep(5)
        print("[INFO] 停止 PMD2 监控")
        pmd2_monitor.stop()
        pmd2_monitor.join()
        print("[INFO] PMD2 已安全停止 ✅")

if __name__=="__main__":
    main()
