# Real-Time ESP32 SAC Inference for MuJoCo CartPole (HIL)

A Hardware-in-the-Loop (HIL) control system executing a quantized Soft Actor-Critic (SAC) neural network in deterministic real-time on an ESP32 (ESP-IDF & FreeRTOS) to swing up and stabilize a simulated CartPole environment in MuJoCo over UART.

## Key Features

* **Deterministic Dual-Core Execution:** Strict task isolation using FreeRTOS core pinning to decouple serial I/O from neural network math.
* **Custom Quantized Inference with Native Normalization:** Standalone C++ inference engine (`int8_t` weights with pre-scaled pointer arithmetic and pruning)—zero external ML library overhead. Raw physical sensor data is normalized directly on the MCU using statically compiled `VecNormalize` parameters (Mean/Std).
* **Robust RL Training Pipeline:** The SAC agent is trained using Squared Exponential (Gaussian RBF) reward shaping and true OS-level domain randomization, allowing the MCU to recover the pendulum from extreme, non-deterministic starting angles.
* **Maximum Hardware Utilization:** Enforced 240 MHz CPU clock, 80 MHz flash read speed, and hardware compiler optimization (`-O3`) via custom `sdkconfig.defaults` to meet strict <10 ms control deadlines.
* **Zero-Blocking RAM Logging:** Custom `esp_log_set_vprintf` hook routing logs to a 4 KB FreeRTOS Stream Buffer in SRAM, eliminating UART bottlenecks and heap fragmentation during control loops.
* **Automated HIL Benchmarking:** Python-based profiling suite that extracts real-time telemetry and generates comprehensive standalone HTML reports. It calculates the *true* hardware quantization MAE by evaluating the live ESP32 INT8 action against an uncompressed FP32 Python reference model on the fly.

---

## System Architecture

<img width="100%" alt="SystemArchitecture" src="https://github.com/user-attachments/assets/32764aa0-104b-4ce1-9fcd-a2ae53a642c6" />

### Real-Time Design Choices

* **Core Pinning:** `CommsTask` handles UART interrupts, string parsing, and telemetry on **Core 0**, while `InferenceTask` runs undisturbed on **Core 1**.
* **Depth-1 Queues:** `obs_queue` and `action_queue` are initialized with a depth of `1` to enforce tight lockstep synchronization between the physics engine and the microcontroller without stale state buildup.
* **Static Memory Allocation:** All buffers and queues reside statically in SRAM or on task stacks (`temp_buf[256]`), preventing runtime `malloc`/`free` overhead and heap fragmentation.
* **Dynamic Telemetry (`GET_META`):** On-demand extraction of FreeRTOS Stack High Water marks, SRAM/Heap availability, and CPU configurations without disrupting the real-time inference loop.

---

## Hardware Benchmark & Profiling

The repository includes a custom Python benchmarking tool (`benchmark.py`) that executes a full Hardware-in-the-Loop test against the ESP32 and generates a dark-themed HTML report.

[View interactive HTML Benchmark Report](https://eliashaering.github.io/InvertedPendulum/simulation/benchmark_report.html?v=4)

