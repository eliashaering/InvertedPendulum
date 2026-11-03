import time
import serial
import struct
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

from mltraining.train_sac import MuJoCoCartPoleEnv 

# ==========================================
# CONFIGURATION
# ==========================================
USE_ESP32 = True  # True: ESP32 hardware inference | False: Local Python inference
SERIAL_PORT = '/dev/cu.usbserial-3120' # Adjust to match your system's port

# UART Binary Protocol Definitions
UART_Header_OBS = 0x00
UART_Header_ACT = 0x01
UART_Header_LOG = 0x02
UART_Header_META = 0x03
SCALE = 1000.0
# ==========================================

env = MuJoCoCartPoleEnv()

if USE_ESP32:
    esp32 = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    time.sleep(2) # Wait for ESP32 to reboot and initialize
    esp32.reset_input_buffer() # Clear boot logs before starting control loop
else:
    model = SAC.load("mltraining/models/mujoco_cartpole_sac")

obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        # 1. Determine action based on selected backend
        if USE_ESP32:
            # Pack observation packet (<B4h = 1 uint8 + 4 int16 = 9 Bytes)
            payload = struct.pack('<B4h', 
                UART_Header_OBS,
                int(obs[0] * SCALE),
                int(obs[1] * SCALE),
                int(obs[2] * SCALE),
                int(obs[3] * SCALE)
            )
            esp32.write(payload)
            
            # Read exactly 3 Bytes for the Action packet (<Bh = 1 uint8 + 1 int16)
            raw_resp = esp32.read(3)
            
            action = np.array([0.0]) # Default fallback action
    
            if len(raw_resp) == 3:
                header, raw_action = struct.unpack('<Bh', raw_resp)
                
                if header == UART_Header_ACT:
                    action_value = raw_action / SCALE
                    action = np.array([action_value])
                else:
                    # Sync error, clear buffer
                    print(f"Packet Error: Received unknown header {header}")
                    esp32.reset_input_buffer()
            else:
                # Timeout occurred
                print("Serial read timeout")
                esp32.reset_input_buffer()
                
        else:
            action, _ = model.predict(obs, deterministic=True)
            print(f"SIM: Executed action: {action}")
        
        # 2. Execute physics step
        obs, reward, terminated, truncated, _ = env.step(action)
        viewer.sync()
        
        if terminated or truncated:
            obs, _ = env.reset()
            
        # 3. Enforce real-time execution step rate
        time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
