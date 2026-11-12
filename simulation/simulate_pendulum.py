import time
import serial
import struct
import numpy as np
import mujoco
import mujoco.viewer
import sys
import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

env = make_vec_env(MuJoCoCartPoleEnv, n_envs=1)
env = VecNormalize.load("mltraining/models/vec_normalize.pkl", env)
env.training = False    
env.norm_reward = False

if USE_ESP32:
    esp32 = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    time.sleep(2) # Wait for ESP32 to reboot and initialize
    esp32.reset_input_buffer() # Clear boot logs before starting control loop
else:
    model = SAC.load("mltraining/models/mujoco_cartpole_sac")

obs = env.reset()
base_env = env.envs[0].unwrapped


with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        # 1. Determine action based on selected backend
        if USE_ESP32:
            # Pack observation packet (<B4h = 1 uint8 + 4 int16 = 9 Bytes)
            raw_obs = env.get_original_obs()[0]
            
            payload = struct.pack('<B4h', 
                UART_Header_OBS,
                int(raw_obs[0] * SCALE),
                int(raw_obs[1] * SCALE),
                int(raw_obs[2] * SCALE),
                int(raw_obs[3] * SCALE)
            )
            esp32.write(payload)
            
            # Read exactly 3 Bytes for the Action packet (<Bh = 1 uint8 + 1 int16)
            raw_resp = esp32.read(3)
            
            action = np.array([[0.0]]) # Default fallback action
    
            if len(raw_resp) == 3:
                header, raw_action = struct.unpack('<Bh', raw_resp)
                
                if header == UART_Header_ACT:
                    action_value = raw_action / SCALE
                    action = np.array([[action_value]])
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
        obs, rewards, dones, infos = env.step(action)
        viewer.sync()
        
        
        #if terminated or truncated:
        #    obs, _ = env.reset()
            
        # 3. Enforce real-time execution step rate
        time_until_next_step = base_env.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
