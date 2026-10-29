# Real-Time ESP32 SAC Inference for MuJoCo CartPole (HIL)

A Hardware-in-the-Loop (HIL) control system executing a quantized Soft Actor-Critic (SAC) neural network in deterministic real-time on an ESP32 (ESP-IDF & FreeRTOS) to swing up and stabilize a simulated CartPole environment in MuJoCo over UART.

## Key Features

* **Deterministic Dual-Core Execution:** Strict task isolation using FreeRTOS core pinning to decouple serial I/O from neural network math.
* **Custom Quantized Inference:** Standalone C++ inference engine (`int8_t` weights with pre-scaled pointer arithmetic and pruning)—zero external ML library overhead.
* **Zero-Blocking RAM Logging:** Custom `esp_log_set_vprintf` hook routing logs to a 4 KB FreeRTOS Stream Buffer in SRAM, eliminating UART bottlenecks and heap fragmentation during control loops.
* **Microsecond Profiling:** Cycle-accurate latency tracking via `esp_timer` to guarantee strict adherence to simulation timesteps.

---

## System Architecture

<img width="100%" alt="SystemArchitecture" src="https://github.com/user-attachments/assets/32764aa0-104b-4ce1-9fcd-a2ae53a642c6" />

### Real-Time Design Choices

* **Core Pinning:** `CommsTask` handles UART interrupts and string parsing on **Core 0**, while `InferenceTask` runs undisturbed on **Core 1**.
* **Depth-1 Queues:** `obs_queue` and `action_queue` are initialized with a depth of `1` to enforce tight lockstep synchronization between the physics engine and the microcontroller without stale state buildup.
* **Static Memory Allocation:** All buffers and queues reside statically in SRAM or on task stacks (`temp_buf[256]`), preventing runtime `malloc`/`free` overhead and heap fragmentation.


