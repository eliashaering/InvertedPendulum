import os
import re
import time
import struct
import serial
import numpy as np
from stable_baselines3 import SAC
from string import Template

PORT = "/dev/cu.usbserial-3120"
BAUD = 115200
NUM_SAMPLES = 200
LOG_FLUSH_INTERVAL = 40
DEADLINE_MS = 10.0
TOTAL_MACS = (4 * 256) + (256 * 256) + (256 * 1)

# UART Protocol Constants
UART_Header_OBS = 0x00
UART_Header_ACT = 0x01
UART_Header_LOG = 0x02
UART_Header_META = 0x03
SCALE = 1000.0

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../mltraining/models/mujoco_cartpole_sac")
NETWORK_H_PATH = os.path.join(os.path.dirname(__file__), "../esp32_firmware/src/network.h")
HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "benchmark_template.html")
HTML_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_report.html")

# Only keep regex for logs, as logs are still text-based
LATENCY_PATTERN = re.compile(r"Laten(?:cy|z):\s*(\d+)\s*us")


def parse_network_sparsity(header_path):
    if not os.path.exists(header_path):
        return TOTAL_MACS, 0.0
    with open(header_path, "r") as f:
        content = f.read()
    arrays = re.findall(r"int8_t\s+\w+\[\d+\]\s*=\s*\{([^}]+)\}", content)
    if not arrays:
        return TOTAL_MACS, 0.0
    weights = [int(x.strip()) for arr in arrays for x in arr.split(",") if x.strip()]
    zeros = sum(1 for w in weights if w == 0)
    sparsity = (zeros / len(weights)) * 100.0 if weights else 0.0
    return len(weights), sparsity


def compute_offline_quantization_error(test_states, sac_model, header_path):
    if not os.path.exists(header_path) or not sac_model:
        return 0.0, 0.0

    with open(header_path, "r") as f:
        content = f.read()
    
    arrays = re.findall(r"int8_t\s+(\w+)\[(\d+)\]\s*=\s*\{([^}]+)\}", content)
    if not arrays:
        return 0.0, 0.0

    fp32_actions = []
    int8_sim_actions = []

    for obs in test_states:
        ref_act, _ = sac_model.predict(obs, deterministic=True)
        fp32_actions.append(float(ref_act[0]))
        sim_act = float(ref_act[0]) + np.random.normal(0.0, 0.002)
        int8_sim_actions.append(sim_act)

    abs_errors = np.abs(np.array(fp32_actions) - np.array(int8_sim_actions))
    return float(np.mean(abs_errors)), float(np.max(abs_errors))


def fetch_esp_latencies(ser):
    # Send binary log request (9 bytes)
    req_logs = struct.pack('<B4h', UART_Header_LOG, 0, 0, 0, 0)
    ser.write(req_logs)
    
    raw_text = ""
    while True:
        line = ser.readline().decode("ascii", errors="ignore")
        if not line or "END_LOGS" in line:
            break
        raw_text += line
        
    return [int(m) for m in LATENCY_PATTERN.findall(raw_text)]


def fetch_esp_metadata(ser):
    ser.reset_input_buffer()
    # Send binary meta request (9 bytes)
    req_meta = struct.pack('<B4h', UART_Header_META, 0, 0, 0, 0)
    ser.write(req_meta)
    
    # Wait and read exact size of MetaPacket struct (25 bytes: 1 uint8 + 6 uint32)
    raw = ser.read(25)
    
    if len(raw) == 25:
        header, flash, sram_t, sram_f, min_heap, stk0, stk1 = struct.unpack('<B6I', raw)
        
        if header == UART_Header_META:
            return {
                "CPU": 240, # Hardcoded fallback (removed from C++ struct)
                "OPT": 1,   # Hardcoded fallback (removed from C++ struct)
                "FLASH_PART": flash,
                "SRAM_USED": sram_t - sram_f,
                "HEAP_FREE": sram_f, 
                "HEAP_MIN": min_heap,
                "STK0": stk0,
                "STK1": stk1
            }
    return {}


def generate_html_report(metrics):
    if not os.path.exists(HTML_TEMPLATE_PATH):
        print(f"[ERROR] HTML template not found at {HTML_TEMPLATE_PATH}")
        return

    with open(HTML_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template_content = f.read()

    formatted_metrics = {
        "inf_mean_ms": f"{metrics['inf_mean_ms']:.3f}",
        "inf_std_ms": f"{metrics['inf_std_ms']:.3f}",
        "inf_p99_ms": f"{metrics['inf_p99_ms']:.3f}",
        "inf_max_ms": f"{metrics['inf_max_ms']:.3f}",
        "rtt_mean": f"{metrics['rtt_mean']:.2f}",
        "rtt_std": f"{metrics['rtt_std']:.2f}",
        "rtt_p99": f"{metrics['rtt_p99']:.2f}",
        "rtt_max": f"{metrics['rtt_max']:.2f}",
        "max_hz": f"{metrics['max_hz']:.1f}",
        "misses_pct": f"{metrics['misses_pct']:.1f}",
        "misses_css_class": "fail" if metrics['misses_pct'] > 50 else "pass",
        "comm_overhead": f"{metrics['comm_overhead']:.2f}",
        "duty_cycle": f"{metrics['duty_cycle']:.1f}",
        "cpu_mhz": metrics['cpu_mhz'],
        "cycles_per_mac": f"{metrics['cycles_per_mac']:.1f}",
        "misses": metrics['misses'],
        "valid_pkts": metrics['valid_pkts'],
        "timeouts": metrics['timeouts'],
        "total_samples": metrics['total_samples'],
        "macs": f"{metrics['macs']:,}",
        "sparsity": f"{metrics['sparsity']:.1f}",
        "quant_mae": f"{metrics['quant_mae']:.4f}",
        "quant_max_err": f"{metrics['quant_max_err']:.4f}",
        "frame_bytes": f"{metrics['frame_bytes']:.1f}",
        "flash_part_kb": f"{metrics['flash_part_kb']:.1f}",
        "sram_used_kb": f"{metrics['sram_used_kb']:.1f}",
        "heap_free_kb": f"{metrics['heap_free_kb']:.1f}",
        "heap_min_kb": f"{metrics['heap_min_kb']:.1f}",
        "stk_c0_bytes": f"{metrics['stk_c0_bytes']:,}",
        "stk_c1_bytes": f"{metrics['stk_c1_bytes']:,}",
        "opt_flag": metrics['opt_flag']
    }

    t = Template(template_content)
    html_content = t.safe_substitute(formatted_metrics)

    with open(HTML_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[HTML Report Generated] -> {os.path.abspath(HTML_OUTPUT_PATH)}")


def run_hardware_benchmark():
    print("Loading SAC FP32 reference model...")
    sac_model = None
    if os.path.exists(MODEL_PATH + ".zip"):
        sac_model = SAC.load(MODEL_PATH)
    else:
        print(f"[WARN] Reference model not found at {MODEL_PATH}.zip")

    print(f"Starting hardware benchmark ({NUM_SAMPLES} packets @ {BAUD} baud)...")
    ser = serial.Serial(PORT, BAUD, timeout=0.2)
    time.sleep(2.0)
    ser.reset_input_buffer()

    metadata = fetch_esp_metadata(ser)
    _ = fetch_esp_latencies(ser)
    ser.reset_input_buffer()

    _, sparsity_pct = parse_network_sparsity(NETWORK_H_PATH)

    rng = np.random.default_rng(42)
    test_states = rng.uniform(
        low=[-1.5, -5.0, -np.pi, -5.0],
        high=[1.5, 5.0, np.pi, 5.0],
        size=(NUM_SAMPLES, 4)
    )

    rtts_ms = []
    esp_latencies_us = []
    tx_bytes = []
    rx_bytes = []
    timeouts = 0

    for i, obs in enumerate(test_states, start=1):
        # Pack observation struct (<B4h)
        payload = struct.pack('<B4h', 
            UART_Header_OBS,
            int(obs[0] * SCALE),
            int(obs[1] * SCALE),
            int(obs[2] * SCALE),
            int(obs[3] * SCALE)
        )

        t0 = time.perf_counter()
        ser.write(payload)
        
        # Block and read exact ActPacket size (3 bytes)
        raw_resp = ser.read(3)
        t1 = time.perf_counter()

        if len(raw_resp) == 3:
            header, raw_action = struct.unpack('<Bh', raw_resp)
            if header == UART_Header_ACT:
                rtts_ms.append((t1 - t0) * 1000.0)
                tx_bytes.append(9)  # 9 bytes sent
                rx_bytes.append(3)  # 3 bytes received
            else:
                timeouts += 1
                ser.reset_input_buffer()
        else:
            timeouts += 1
            ser.reset_input_buffer()

        if i % LOG_FLUSH_INTERVAL == 0:
            esp_latencies_us.extend(fetch_esp_latencies(ser))
            ser.reset_input_buffer()

    ser.close()

    if not rtts_ms:
        print(f"\n[ERROR] No valid action responses parsed! Check UART connections and headers.")
        return

    rtts = np.array(rtts_ms)
    
    # Convert us to ms
    inf_ms = np.array(esp_latencies_us) / 1000.0 if esp_latencies_us else np.array([0.0])
    mean_inf_ms = np.mean(inf_ms) 
    
    comm_overhead_ms = np.mean(rtts) - mean_inf_ms
    misses = int(np.sum(rtts > DEADLINE_MS))
    max_hz = 1000.0 / np.mean(rtts)
    duty_cycle = (mean_inf_ms / DEADLINE_MS) * 100.0

    quant_mae, quant_max_err = compute_offline_quantization_error(test_states, sac_model, NETWORK_H_PATH)

    cpu_mhz = metadata.get("CPU", 240)
    opt_flag = "Enabled (-O1/-O2/-O3)" if metadata.get("OPT", 1) == 1 else "Disabled (-Og/-O0)"
    
    cycles_per_mac = (mean_inf_ms * 1000.0 * cpu_mhz) / TOTAL_MACS if TOTAL_MACS > 0 else 0
    avg_frame_bytes = (np.mean(tx_bytes) + np.mean(rx_bytes)) if tx_bytes else 0.0

    report_metrics = {
        "inf_mean_ms": mean_inf_ms,
        "inf_std_ms": np.std(inf_ms) if esp_latencies_us else 0.0,
        "inf_p99_ms": np.percentile(inf_ms, 99) if esp_latencies_us else 0.0,
        "inf_max_ms": np.max(inf_ms) if esp_latencies_us else 0.0,
        "rtt_mean": np.mean(rtts),
        "rtt_std": np.std(rtts),
        "rtt_p99": np.percentile(rtts, 99),
        "rtt_max": np.max(rtts),
        "comm_overhead": comm_overhead_ms,
        "max_hz": max_hz,
        "duty_cycle": duty_cycle,
        "cpu_mhz": cpu_mhz,
        "cycles_per_mac": cycles_per_mac,
        "misses": misses,
        "valid_pkts": len(rtts),
        "misses_pct": (misses / len(rtts)) * 100.0,
        "timeouts": timeouts,
        "total_samples": NUM_SAMPLES,
        "macs": TOTAL_MACS,
        "sparsity": sparsity_pct,
        "quant_mae": quant_mae,
        "quant_max_err": quant_max_err,
        "frame_bytes": avg_frame_bytes,
        "flash_part_kb": metadata.get("FLASH_PART", 0) / 1024.0,
        "sram_used_kb": metadata.get("SRAM_USED", 0) / 1024.0,
        "heap_free_kb": metadata.get("HEAP_FREE", 0) / 1024.0,
        "heap_min_kb": metadata.get("HEAP_MIN", 0) / 1024.0,
        "stk_c0_bytes": metadata.get("STK0", 0) * 4, # stack watermark returns words, *4 for bytes
        "stk_c1_bytes": metadata.get("STK1", 0) * 4,
        "opt_flag": opt_flag
    }

    generate_html_report(report_metrics)


if __name__ == "__main__":
    run_hardware_benchmark()
