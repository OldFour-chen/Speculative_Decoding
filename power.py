import pynvml
import torch
import time
import threading


class PowerMonitor:
    def __init__(self, gpu_index=0, poll_interval=0.001):
        """
        poll_interval: 目标采样间隔（秒）。
        注意：Python 线程调度和 NVML 底层更新频率（通常 ~1ms）会限制实际采样率。
        """
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.poll_interval = poll_interval
        self.is_monitoring = False
        self.power_records = []   # 存储 (时间戳, 功率mW)
        self.monitor_thread = None
        self._lock = threading.Lock()           
        self._thread_exception = None          

    def shutdown(self):
        """释放 NVML 资源，应在所有测量结束后调用。"""
        pynvml.nvmlShutdown()                  

    def _record_power(self):
        try:                                   
            while self.is_monitoring:
                ts = time.perf_counter()
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                with self._lock:                # FIX 5: 写入时加锁
                    self.power_records.append((ts, power_mw))
                time.sleep(self.poll_interval)
        except Exception as e:
            self._thread_exception = e
            self.is_monitoring = False        

    def start(self):
        self.power_records = []
        self._thread_exception = None
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._record_power,
            daemon=True                        
        )
        self.monitor_thread.start()

    def stop(self):
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        if self._thread_exception:             
            raise RuntimeError(
                f"PowerMonitor thread failed: {self._thread_exception}"
            ) from self._thread_exception

    def calculate_energy(self):
        """
        使用梯形积分法计算总能耗。
        公式: E = Σ (P_i + P_{i+1}) / 2 * Δt
        """
        with self._lock:                      
            records = list(self.power_records)

        if len(records) < 2:
            return 0.0

        total_energy_joules = 0.0
        for i in range(1, len(records)):
            t0, p0_mw = records[i - 1]
            t1, p1_mw = records[i]
            dt = t1 - t0
            avg_power_w = (p0_mw + p1_mw) / 2.0 / 1000.0
            total_energy_joules += avg_power_w * dt

        return total_energy_joules


def measure_inference_by_integration(gpu_index=0):
    # 1. 准备模型和数据
    model = torch.nn.Linear(4096, 4096).cuda(gpu_index)
    inputs = torch.randn(256, 4096).cuda(gpu_index)

    # 2. 预热（避免测到冷启动的突发功耗）
    print("Warming up...")
    with torch.no_grad():
        for _ in range(20):
            model(inputs)
    torch.cuda.synchronize(gpu_index)

    # 3. 初始化监控器
    monitor = PowerMonitor(gpu_index=gpu_index, poll_interval=0.001)

    try:                                       
        # 4. 开始后台采样
        monitor.start()

        # 5. 执行推理任务
        with torch.no_grad():
            for _ in range(10):
                model(inputs)
        torch.cuda.synchronize(gpu_index)
        # ====================================================

        # 6. 停止采样（内部会 join 线程并检查子线程异常）
        monitor.stop()

    except Exception:
        monitor.is_monitoring = False           # 确保线程能退出
        raise
    finally:
        monitor.shutdown()                     

    # 7. 用采样记录自身的首尾时间戳统计，避免与外部计时偏差
    records = monitor.power_records
    if records:
        time_taken_ms = (records[-1][0] - records[0][0]) * 1000
    else:
        time_taken_ms = 0.0

    energy_j = monitor.calculate_energy()
    sample_count = len(records)

    print("-" * 50)
    print(f"Task Duration      : {time_taken_ms:.2f} ms")
    print(f"Samples Collected  : {sample_count}")
    if sample_count > 1:
        print(f"Actual Sample Rate : {time_taken_ms / sample_count:.2f} ms / sample")
    print(f"Integrated Energy  : {energy_j:.4f} Joules")
    print("-" * 50)


if __name__ == "__main__":
    measure_inference_by_integration(gpu_index=0)